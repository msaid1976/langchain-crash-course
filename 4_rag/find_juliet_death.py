import os
from langchain_community.document_loaders import TextLoader

# Load the Romeo and Juliet text
file_path = os.path.join('books', 'romeo_and_juliet.txt')
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()
text = documents[0].page_content.lower()

# Find sections that mention Juliet and death-related words
lines = text.split('\n')
relevant_sections = []

for i, line in enumerate(lines):
    if 'juliet' in line and any(word in line for word in ['die', 'death', 'poison', 'dagger', 'stab']):
        # Get some context around the line
        start = max(0, i-2)
        end = min(len(lines), i+3)
        context = '\n'.join(lines[start:end]).strip()
        if context and len(context) > 20:
            relevant_sections.append(context)

print(f'Found {len(relevant_sections)} relevant sections about Juliet death:')
for i, section in enumerate(relevant_sections[:3], 1):
    print(f'\n--- Section {i} ---')
    print(section)
    print()

# Also look for the actual death scene
print("\n" + "="*60)
print("SEARCHING FOR JULIET'S ACTUAL DEATH SCENE")
print("="*60)

death_keywords = ['juliet', 'poison', 'dagger', 'tomb', 'vault', 'romeo']
sections_with_multiple_keywords = []

for i, line in enumerate(lines):
    keyword_count = sum(1 for keyword in death_keywords if keyword in line)
    if keyword_count >= 2:  # Line contains at least 2 death-related keywords
        start = max(0, i-5)  # More context
        end = min(len(lines), i+6)
        context = '\n'.join(lines[start:end]).strip()
        if len(context) > 50:
            sections_with_multiple_keywords.append((keyword_count, context))

# Sort by keyword count (most relevant first)
sections_with_multiple_keywords.sort(key=lambda x: x[0], reverse=True)

print(f'Found {len(sections_with_multiple_keywords)} sections with multiple keywords:')
for i, (count, section) in enumerate(sections_with_multiple_keywords[:3], 1):
    print(f'\n--- Section {i} (keywords: {count}) ---')
    print(section)
    print()
