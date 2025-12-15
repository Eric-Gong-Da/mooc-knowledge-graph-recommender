import json
import random
import pickle
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from gensim.models import Word2Vec


class MetaPathRandomWalk:
    def __init__(self, data_dir="relations", checkpoint_dir="checkpoints"):
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.user_watched_video = defaultdict(list)
        self.video_covers_concept = defaultdict(list)
        self.concept_covered_by_video = defaultdict(list)
        self.video_watched_by_user = defaultdict(list)
        
        self._load_relations()
    
    def _load_relations(self):
        print("Loading relations...")
        
        self._load_user_video()
        self._load_video_concept()
    
    def _count_lines(self, file_path):
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
    
    def _load_user_video(self):
        file_path = self.data_dir / "user-video.json"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
        
        total_lines = self._count_lines(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Loading user-video", ncols=100):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    user_id = parts[0].strip()
                    video_id = parts[1].strip()
                    self.user_watched_video[user_id].append(video_id)
                    self.video_watched_by_user[video_id].append(user_id)
    
    def _load_video_concept(self):
        file_path = self.data_dir / "video-concept.json"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
        
        total_lines = self._count_lines(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Loading video-concept", ncols=100):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    video_id = parts[0].strip()
                    concept_id = parts[1].strip()
                    self.video_covers_concept[video_id].append(concept_id)
                    self.concept_covered_by_video[concept_id].append(video_id)
    
    def random_walk(self, start_user, walk_length=8):
        walk = [start_user]
        current_node = start_user
        node_type = 'user'
        
        for step in range(walk_length - 1):
            if node_type == 'user':
                if current_node in self.user_watched_video and self.user_watched_video[current_node]:
                    next_node = random.choice(self.user_watched_video[current_node])
                    walk.append(next_node)
                    current_node = next_node
                    node_type = 'video'
                else:
                    break
            
            elif node_type == 'video':
                if current_node in self.video_covers_concept and self.video_covers_concept[current_node]:
                    next_node = random.choice(self.video_covers_concept[current_node])
                    walk.append(next_node)
                    current_node = next_node
                    node_type = 'concept'
                else:
                    break
            
            elif node_type == 'concept':
                if current_node in self.concept_covered_by_video and self.concept_covered_by_video[current_node]:
                    next_node = random.choice(self.concept_covered_by_video[current_node])
                    walk.append(next_node)
                    current_node = next_node
                    node_type = 'video'
                else:
                    break
        
        return walk
    
    def generate_walks(self, num_walks_per_user=10, walk_length=8, checkpoint_interval=1000, resume=True):
        checkpoint_file = self.checkpoint_dir / f"walks_checkpoint_{num_walks_per_user}_{walk_length}.pkl"
        
        users = list(self.user_watched_video.keys())
        print(f"Found {len(users)} users")
        
        walks = []
        start_idx = 0
        
        if resume and checkpoint_file.exists():
            print(f"Loading checkpoint from {checkpoint_file}...")
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                walks = checkpoint_data['walks']
                start_idx = checkpoint_data['user_idx']
            print(f"Resumed from user {start_idx}/{len(users)}, {len(walks)} walks loaded")
        
        try:
            for i in tqdm(range(start_idx, len(users)), desc="Generating walks", ncols=100, initial=start_idx, total=len(users)):
                user = users[i]
                for _ in range(num_walks_per_user):
                    walk = self.random_walk(user, walk_length)
                    if len(walk) > 1:
                        walks.append([str(node) for node in walk])
                
                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint_file, walks, i + 1)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving checkpoint...")
            self._save_checkpoint(checkpoint_file, walks, i)
            print(f"Checkpoint saved. Resume by running again.")
            raise
        
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"Completed! Removed checkpoint file.")
        
        return walks
    
    def _save_checkpoint(self, checkpoint_file, walks, user_idx):
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'walks': walks,
                'user_idx': user_idx
            }, f)
        print(f"\nCheckpoint saved: {len(walks)} walks, user {user_idx}")
    
    def train_embeddings(self, walks, embedding_dim=128, window=5, min_count=1, workers=4):
        print(f"\nTraining Word2Vec model with {len(walks)} walks...")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Window size: {window}")
        print(f"Workers: {workers}")
        
        from gensim.models.callbacks import CallbackAny2Vec
        
        class TqdmCallback(CallbackAny2Vec):
            def __init__(self, epochs):
                self.epochs = epochs
                self.pbar = None
            
            def on_train_begin(self, model):
                self.pbar = tqdm(total=self.epochs, desc="Training epochs", ncols=100)
            
            def on_epoch_end(self, model):
                if self.pbar:
                    self.pbar.update(1)
            
            def on_train_end(self, model):
                if self.pbar:
                    self.pbar.close()
        
        callback = TqdmCallback(epochs=10)
        
        model = Word2Vec(
            sentences=walks,
            vector_size=embedding_dim,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1,
            epochs=10,
            callbacks=[callback]
        )
        
        return model
    
    def save_user_embeddings(self, model, output_file="user_embeddings.txt"):
        users = list(self.user_watched_video.keys())
        
        print(f"Saving embeddings for {len(users)} users to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for user in tqdm(users, desc="Saving embeddings", ncols=100):
                if user in model.wv:
                    embedding = model.wv[user]
                    embedding_str = ' '.join([f"{val:.6f}" for val in embedding])
                    f.write(f"{user}\t{embedding_str}\n")
        
        print(f"Saved embeddings to {output_file}")


def filter_users_by_chapter(walker, chapter_prefix="02"):
    filtered_user_watched = defaultdict(list)
    
    course_file = Path("entities/course.json")
    video_to_chapter = {}
    
    if course_file.exists():
        print(f"Loading course data to filter chapter {chapter_prefix}...")
        with open(course_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading courses", ncols=100):
                line = line.strip()
                if not line:
                    continue
                try:
                    course = json.loads(line)
                    if 'video_order' in course and 'chapter' in course:
                        for vid, chap in zip(course['video_order'], course['chapter']):
                            if chap and str(chap).startswith(chapter_prefix):
                                video_to_chapter[vid] = chap
                except:
                    continue
        
        print(f"Found {len(video_to_chapter)} videos in chapter {chapter_prefix}")
        
        for user, videos in tqdm(walker.user_watched_video.items(), desc="Filtering users", ncols=100):
            chapter_videos = [v for v in videos if v in video_to_chapter]
            if chapter_videos:
                filtered_user_watched[user] = videos
        
        print(f"Filtered to {len(filtered_user_watched)} users who watched chapter {chapter_prefix} videos")
        return filtered_user_watched
    else:
        print(f"Course file not found, using all users")
        return walker.user_watched_video


def main(test_mode=False, limit_users=None, chapter_filter=None):
    walker = MetaPathRandomWalk(data_dir="relations")
    
    if chapter_filter:
        print(f"\n{'='*80}")
        print(f"CHAPTER FILTER MODE: Only users who watched chapter {chapter_filter} videos")
        print(f"{'='*80}\n")
        
        walker.user_watched_video = filter_users_by_chapter(walker, chapter_filter)
    
    if test_mode and limit_users:
        print(f"\n{'='*80}")
        print(f"TEST MODE: Processing first {limit_users} users only")
        print(f"{'='*80}\n")
        
        all_users = list(walker.user_watched_video.keys())
        limited_users = all_users[:limit_users]
        walker.user_watched_video = {k: walker.user_watched_video[k] for k in limited_users if k in walker.user_watched_video}
    
    print("\n" + "="*80)
    print("Configuration Options:")
    print("="*80)
    print("\nOption 1: Quick Test (Small Scale)")
    print("  - num_walks_per_user: 5")
    print("  - walk_length: 6")
    print("  - Estimated time: ~5-10 minutes")
    print("  - Effect: Basic embeddings, good for testing")
    
    print("\nOption 2: Balanced (Recommended)")
    print("  - num_walks_per_user: 10")
    print("  - walk_length: 8")
    print("  - Estimated time: ~15-30 minutes")
    print("  - Effect: Good quality embeddings for most tasks")
    
    print("\nOption 3: High Quality")
    print("  - num_walks_per_user: 20")
    print("  - walk_length: 10")
    print("  - Estimated time: ~1-2 hours")
    print("  - Effect: High quality embeddings, captures more patterns")
    
    print("\nOption 4: Production Grade")
    print("  - num_walks_per_user: 50")
    print("  - walk_length: 12")
    print("  - Estimated time: ~3-5 hours")
    print("  - Effect: Best quality, captures comprehensive patterns")
    
    print("\n" + "="*80)
    if chapter_filter:
        print(f"Current Configuration: Chapter {chapter_filter} Filter Mode")
    elif test_mode:
        print("Current Configuration: TEST MODE")
        print(f"Limited to {limit_users} users")
    else:
        print("Current Configuration: Option 2 (Balanced)")
    print("="*80 + "\n")
    
    num_walks_per_user = 50
    walk_length = 12
    embedding_dim = 128
    
    walks = walker.generate_walks(
        num_walks_per_user=num_walks_per_user,
        walk_length=walk_length,
        checkpoint_interval=1000,
        resume=True
    )
    
    print(f"\nGenerated {len(walks)} walks")
    
    model = walker.train_embeddings(
        walks,
        embedding_dim=embedding_dim
    )
    
    output_suffix = ""
    if chapter_filter:
        output_suffix = f"_chapter{chapter_filter}"
    elif test_mode:
        output_suffix = "_test"
    
    output_file = f"data/user_embeddings_128d{output_suffix}.txt"
    
    walker.save_user_embeddings(
        model,
        output_file=output_file
    )
    
    print(f"\nDone! Output saved to {output_file}")


if __name__ == "__main__":
    import sys
    
    test_mode = False
    limit_users = None
    chapter_filter = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_mode = True
            limit_users = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        elif sys.argv[1] == "--chapter":
            chapter_filter = sys.argv[2] if len(sys.argv) > 2 else "03"
    
    main(test_mode=test_mode, limit_users=limit_users, chapter_filter=chapter_filter)