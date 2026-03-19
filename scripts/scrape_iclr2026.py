#%%
import openreview
import openreview.tools
import os
import json
from collections import Counter, defaultdict
from dotenv import load_dotenv

load_dotenv()

client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username=os.getenv('OPENREVIEW_USERNAME'),
    password=os.getenv('OPENREVIEW_PASSWORD'),
)

venue_id = 'ICLR.cc/2026/Conference'

#%%
# Step 1: Get all submissions
print("Fetching submissions...")
submissions = client.get_all_notes(
    invitation=f'{venue_id}/-/Submission',
)
print(f"Found {len(submissions)} submissions")

#%%
# Step 2: Build paper list with basic info
papers = {}
for s in submissions:
    content = s.content
    fid = s.forum
    papers[fid] = {
        'id': fid,
        'title': content.get('title', {}).get('value', ''),
        'abstract': content.get('abstract', {}).get('value', ''),
        'number': s.number,
        'scores': [],
        'decision': '',
    }

print(f"Extracted {len(papers)} papers")

#%%
# Step 3: Fetch reviews per paper using concurrent_requests
def fetch_reviews(forum_id):
    try:
        notes = client.get_all_notes(forum=forum_id)
        scores = []
        decision = ''
        for n in notes:
            inv_str = str(n.invitations)
            if 'Official_Review' in inv_str:
                rating = n.content.get('rating', {}).get('value', None)
                if rating is not None:
                    scores.append(rating)
            elif 'Decision' in inv_str:
                decision = n.content.get('decision', {}).get('value', '')
        return forum_id, scores, decision
    except Exception as e:
        return forum_id, [], ''

forum_ids = list(papers.keys())
print(f"Fetching reviews for {len(forum_ids)} papers (parallel)...")

results = openreview.tools.concurrent_requests(
    fetch_reviews,
    forum_ids,
    desc='Fetching reviews',
)

#%%
# Step 4: Merge results
for forum_id, scores, decision in results:
    if forum_id in papers:
        papers[forum_id]['scores'] = scores
        papers[forum_id]['decision'] = decision

papers_list = list(papers.values())

#%%
# Step 5: Save
os.makedirs('data', exist_ok=True)
with open('data/iclr2026_scraped.json', 'w') as f:
    json.dump(papers_list, f, indent=2)

# Summary
has_scores = [p for p in papers_list if p['scores']]
has_decision = [p for p in papers_list if p['decision']]
print(f"\nTotal papers: {len(papers_list)}")
print(f"Papers with scores: {len(has_scores)}")
print(f"Papers with decisions: {len(has_decision)}")
print(f"\nDecision breakdown:")
for dec, count in Counter(p['decision'] for p in papers_list if p['decision']).most_common():
    print(f"  {dec}: {count}")
