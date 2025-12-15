import numpy as np
import json
import random
from pathlib import Path
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


class CourseRecommender:
    def __init__(self, embedding_file, user_course_file, course_info_file):
        self.embedding_file = embedding_file
        self.user_course_file = user_course_file
        self.course_info_file = course_info_file
        
        self.user_embeddings = {}
        self.user_courses = defaultdict(set)
        self.course_names = {}
        
        self._load_data()
    
    def _load_data(self):
        print("Loading user embeddings...")
        with open(self.embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    user_id = parts[0]
                    embedding = np.array([float(x) for x in parts[1].split()])
                    self.user_embeddings[user_id] = embedding
        
        print(f"Loaded {len(self.user_embeddings)} user embeddings")
        
        print("Loading user-course relations...")
        with open(self.user_course_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    user_id = parts[0]
                    course_id = parts[1]
                    self.user_courses[user_id].add(course_id)
        
        print(f"Loaded courses for {len(self.user_courses)} users")
        
        print("Loading course information...")
        with open(self.course_info_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    course = json.loads(line)
                    course_id = course.get('id', '')
                    course_name = course.get('name', 'Unknown')
                    self.course_names[course_id] = course_name
                except:
                    continue
        
        print(f"Loaded {len(self.course_names)} course names")
    
    def get_eligible_users(self, min_courses=7):
        eligible = []
        for user_id, courses in self.user_courses.items():
            if user_id in self.user_embeddings and len(courses) >= min_courses:
                eligible.append(user_id)
        return eligible
    
    def recommend_courses(self, target_user_id, k=10, top_n=10):
        if target_user_id not in self.user_embeddings:
            print(f"User {target_user_id} not found in embeddings")
            return []
        
        target_embedding = self.user_embeddings[target_user_id]
        target_courses = self.user_courses[target_user_id]
        
        all_user_ids = list(self.user_embeddings.keys())
        all_embeddings = np.array([self.user_embeddings[uid] for uid in all_user_ids])
        
        knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        knn.fit(all_embeddings)
        
        distances, indices = knn.kneighbors([target_embedding])
        
        course_scores = defaultdict(float)
        
        for idx, distance in zip(indices[0][1:], distances[0][1:]):
            neighbor_id = all_user_ids[idx]
            neighbor_courses = self.user_courses[neighbor_id]
            
            similarity = 1 - distance
            
            for course_id in neighbor_courses:
                if course_id not in target_courses:
                    course_scores[course_id] += similarity
        
        sorted_courses = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for course_id, score in sorted_courses[:top_n]:
            course_name = self.course_names.get(course_id, course_id)
            recommendations.append({
                'course_id': course_id,
                'course_name': course_name,
                'score': score
            })
        
        return recommendations
    
    def demo(self):
        print("\n" + "="*80)
        print("KNN Course Recommendation Demo")
        print("="*80 + "\n")
        
        eligible_users = self.get_eligible_users(min_courses=7)
        
        if not eligible_users:
            print("No eligible users found (users with >= 7 courses)")
            return
        
        target_user = random.choice(eligible_users)
        
        print(f"Selected User: {target_user}")
        print(f"Number of courses taken: {len(self.user_courses[target_user])}")
        print("\n" + "-"*80)
        print("Courses taken by this user:")
        print("-"*80)
        
        for i, course_id in enumerate(self.user_courses[target_user], 1):
            course_name = self.course_names.get(course_id, course_id)
            print(f"{i}. {course_name}")
        
        print("\n" + "-"*80)
        print("Top 10 Recommended Courses (based on KNN):")
        print("-"*80)
        
        recommendations = self.recommend_courses(target_user, k=10, top_n=10)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['course_name']} (score: {rec['score']:.4f})")
        
        print("\n" + "="*80 + "\n")


def main():
    script_dir = Path(__file__).parent.parent
    
    recommender = CourseRecommender(
        embedding_file=str(script_dir / 'data' / 'user_embeddings_128d.txt'),
        user_course_file=str(script_dir / 'relations' / 'user-course.json'),
        course_info_file=str(script_dir / 'entities' / 'course.json')
    )
    
    recommender.demo()


if __name__ == "__main__":
    main()