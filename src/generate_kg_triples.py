import json
import os
from pathlib import Path
from tqdm import tqdm


class KnowledgeGraphTripleGenerator:
    def __init__(self, input_dir="relations", output_dir="kg_triples"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.relation_configs = {
            "user-course": {
                "input_file": "user-course.json",
                "output_file": "user_enrolled_course.txt",
                "relation": "enrolled_in",
                "description": "User enrollment"
            },
            "user-video": {
                "input_file": "user-video.json",
                "output_file": "user_watched_video.txt",
                "relation": "watched_video",
                "description": "User video watching"
            },
            "course-concept": {
                "input_file": "course-concept.json",
                "output_file": "course_teaches_concept.txt",
                "relation": "teaches_concept",
                "description": "Course teaches concept"
            },
            "parent-son": {
                "input_file": "parent-son.json",
                "output_file": "concept_has_subconcept.txt",
                "relation": "has_subconcept",
                "description": "Concept hierarchy"
            },
            "prerequisite-dependency": {
                "input_file": "prerequisite-dependency.json",
                "output_file": "concept_prerequisite_of.txt",
                "relation": "prerequisite_of",
                "description": "Prerequisite dependency"
            },
            "video-concept": {
                "input_file": "video-concept.json",
                "output_file": "video_covers_concept.txt",
                "relation": "covers_concept",
                "description": "Video covers concept"
            }
        }
    
    def count_lines(self, file_path):
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
    
    def parse_relation_file(self, input_file, relation_name):
        triples = []
        file_path = self.input_dir / input_file
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            return triples
        
        total_lines = self.count_lines(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc=f"Processing {relation_name}", ncols=100):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    head = parts[0].strip()
                    tail = parts[1].strip()
                    triples.append((head, tail))
        
        return triples
    
    def write_triples(self, triples, output_file, relation):
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for head, tail in tqdm(triples, desc=f"Writing {output_file}", ncols=100):
                f.write(f"{head}\t{relation}\t{tail}\n")
        
        return len(triples)
    
    def generate_all_triples(self):
        print("=" * 80)
        print("🚀 Generating Knowledge Graph Triples")
        print("=" * 80)
        print()
        
        stats = {}
        
        for rel_key, config in self.relation_configs.items():
            print(f"📊 Processing: {config['description']} ({config['relation']})")
            print(f"   Input file: {config['input_file']}")
            print(f"   Output file: {config['output_file']}")
            print()
            
            triples = self.parse_relation_file(config['input_file'], config['description'])
            
            if triples:
                count = self.write_triples(triples, config['output_file'], config['relation'])
                stats[rel_key] = {
                    'count': count,
                    'description': config['description'],
                    'relation': config['relation']
                }
                print(f"✅ Generated {count:,} triples")
            else:
                stats[rel_key] = {
                    'count': 0,
                    'description': config['description'],
                    'relation': config['relation']
                }
                print(f"⚠️  No triples generated (file empty or not found)")
            
            print()
        
        self.print_summary(stats)
    
    def print_summary(self, stats):
        print("=" * 80)
        print("📈 Generation Summary")
        print("=" * 80)
        print()
        
        total_triples = 0
        
        for rel_key, info in stats.items():
            count = info['count']
            desc = info['description']
            relation = info['relation']
            total_triples += count
            
            status = "✅" if count > 0 else "⚠️ "
            print(f"{status} {desc:25s} ({relation:20s}): {count:12,} triples")
        
        print()
        print("-" * 80)
        print(f"💾 Total: {total_triples:,} triples")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print("=" * 80)


def main():
    generator = KnowledgeGraphTripleGenerator(
        input_dir="relations",
        output_dir="kg_triples"
    )
    
    generator.generate_all_triples()


if __name__ == "__main__":
    main()