import numpy as np
import json
import random
from pathlib import Path
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


class KNN_RF_CourseRecommender:
    def __init__(self, user_embedding_file, course_embedding_file, user_course_file, course_info_file):
        self.user_embedding_file = user_embedding_file
        self.course_embedding_file = course_embedding_file
        self.user_course_file = user_course_file
        self.course_info_file = course_info_file
        
        self.user_embeddings = {}
        self.course_embeddings = {}
        self.user_courses = defaultdict(set)
        self.course_names = {}
        self.rf_model = None
        
        self._load_data()
        self._train_rf_model()
    
    def _load_data(self):
        print("Loading user embeddings...")
        with open(self.user_embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    user_id = parts[0]
                    embedding = np.array([float(x) for x in parts[1].split()])
                    self.user_embeddings[user_id] = embedding
        
        print(f"Loaded {len(self.user_embeddings)} user embeddings")
        
        print("Loading course embeddings...")
        with open(self.course_embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    course_id = parts[0]
                    embedding = np.array([float(x) for x in parts[1].split()])
                    self.course_embeddings[course_id] = embedding
        
        print(f"Loaded {len(self.course_embeddings)} course embeddings")
        
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
    
    def _train_rf_model(self):
        """
        Train Random Forest model on user-course interactions
        Positive samples: courses user enrolled in
        Negative samples: random courses user didn't enroll in
        """
        print("\n" + "="*80)
        print("Training Random Forest Model")
        print("="*80)
        
        X_train = []
        y_train = []
        
        all_courses_with_embeddings = list(self.course_embeddings.keys())
        
        sample_users = list(self.user_embeddings.keys())[:5000]
        
        print(f"Sampling training data from {len(sample_users)} users...")
        
        for user_id in tqdm(sample_users, desc="Preparing training data", ncols=100):
            if user_id not in self.user_courses:
                continue
            
            user_emb = self.user_embeddings[user_id]
            enrolled_courses = self.user_courses[user_id]
            
            for course_id in enrolled_courses:
                if course_id in self.course_embeddings:
                    course_emb = self.course_embeddings[course_id]
                    feature = np.concatenate([user_emb, course_emb])
                    X_train.append(feature)
                    y_train.append(1)
            
            neg_samples = min(len(enrolled_courses) * 3, 20)
            
            negative_courses = random.sample(all_courses_with_embeddings, 
                                           min(neg_samples, len(all_courses_with_embeddings)))
            
            for course_id in negative_courses:
                if course_id not in enrolled_courses:
                    course_emb = self.course_embeddings[course_id]
                    feature = np.concatenate([user_emb, course_emb])
                    X_train.append(feature)
                    y_train.append(0)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Training data: {len(X_train)} samples ({sum(y_train)} positive, {len(y_train)-sum(y_train)} negative)")
        print("Training Random Forest (this may take a few minutes)...")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        print("Random Forest training completed!")
        print("="*80 + "\n")
    
    def get_eligible_users(self, min_courses=7):
        eligible = []
        for user_id, courses in self.user_courses.items():
            if user_id in self.user_embeddings and len(courses) >= min_courses:
                eligible.append(user_id)
        return eligible
    
    def knn_recall(self, target_user_id, k=10, top_n=50):
        """
        Stage 1: Use KNN to recall candidate courses
        """
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
        
        candidates = []
        for course_id, knn_score in sorted_courses[:top_n]:
            candidates.append({
                'course_id': course_id,
                'knn_score': knn_score
            })
        
        return candidates
    
    def rf_rerank(self, target_user_id, candidates):
        """
        Stage 2: Use Random Forest to rerank candidates
        """
        if target_user_id not in self.user_embeddings:
            return []
        
        user_emb = self.user_embeddings[target_user_id]
        
        X_test = []
        valid_candidates = []
        
        for candidate in candidates:
            course_id = candidate['course_id']
            if course_id in self.course_embeddings:
                course_emb = self.course_embeddings[course_id]
                feature = np.concatenate([user_emb, course_emb])
                X_test.append(feature)
                valid_candidates.append(candidate)
        
        if len(X_test) == 0:
            return []
        
        X_test = np.array(X_test)
        
        rf_probs = self.rf_model.predict_proba(X_test)[:, 1]
        
        for i, candidate in enumerate(valid_candidates):
            candidate['rf_score'] = rf_probs[i]
            candidate['final_score'] = 0.4 * candidate['knn_score'] + 0.6 * candidate['rf_score']
        
        valid_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        return valid_candidates
    
    def recommend_courses(self, target_user_id, k=10, recall_n=50, top_n=10):
        """
        Two-stage recommendation:
        1. KNN recall candidates
        2. RF rerank
        """
        candidates = self.knn_recall(target_user_id, k=k, top_n=recall_n)
        
        if not candidates:
            return []
        
        reranked = self.rf_rerank(target_user_id, candidates)
        
        recommendations = []
        for candidate in reranked[:top_n]:
            course_id = candidate['course_id']
            course_name = self.course_names.get(course_id, course_id)
            recommendations.append({
                'course_id': course_id,
                'course_name': course_name,
                'knn_score': candidate['knn_score'],
                'rf_score': candidate['rf_score'],
                'final_score': candidate['final_score']
            })
        
        return recommendations
    
    def demo(self):
        print("\n" + "="*80)
        print("KNN + Random Forest Course Recommendation Demo")
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
        print("Top 10 Recommended Courses (KNN + Random Forest):")
        print("-"*80)
        
        recommendations = self.recommend_courses(target_user, k=10, recall_n=50, top_n=10)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['course_name']}")
            print(f"   KNN Score: {rec['knn_score']:.4f} | RF Score: {rec['rf_score']:.4f} | Final: {rec['final_score']:.4f}")
        
        print("\n" + "="*80 + "\n")


def main():
    script_dir = Path(__file__).parent.parent
    
    recommender = KNN_RF_CourseRecommender(
        user_embedding_file=str(script_dir / 'data' / 'user_embeddings_128d.txt'),
        course_embedding_file=str(script_dir / 'data' / 'course_embeddings_128d.txt'),
        user_course_file=str(script_dir / 'relations' / 'user-course.json'),
        course_info_file=str(script_dir / 'entities' / 'course.json')
    )
    
    recommender.demo()


if __name__ == "__main__":
    main()
