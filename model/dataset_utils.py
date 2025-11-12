shuffle_mode = False 


def load_dataset(dataset_name, num_examples): 
  print("HI") 

def get_dataloader(dataset): 
  print("HI")

def get_example(): 
  return example

example = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'raw': [
        "[09:15] <alice> does anyone have experience with docker compose?",
        "[09:16] <bob> alice: yeah, what's the issue?",
        "[09:16] <charlie> morning all! quick question about git rebase",
        "[09:17] <alice> bob: my containers keep restarting, not sure why",
        "[09:17] <dave> charlie: shoot, what do you need?",
        "[09:18] <bob> alice: check your logs with docker logs <container_name>",
        "[09:18] <charlie> dave: if I rebase on main, will I lose my commits?",
        "[09:19] <alice> bob: ah it's a memory issue, thanks!",
        "[09:19] <eve> anyone free for code review?",
        "[09:20] <dave> charlie: no, rebase rewrites history but keeps your commits",
        "[09:20] <bob> eve: sure, post the PR link",
        "[09:21] <eve> bob: here you go github.com/project/pr/123"
    ],
    'ascii': [
        "[09:15] <alice> does anyone have experience with docker compose?",
        "[09:16] <bob> alice: yeah, what's the issue?",
        "[09:16] <charlie> morning all! quick question about git rebase",
        "[09:17] <alice> bob: my containers keep restarting, not sure why",
        "[09:17] <dave> charlie: shoot, what do you need?",
        "[09:18] <bob> alice: check your logs with docker logs <container_name>",
        "[09:18] <charlie> dave: if I rebase on main, will I lose my commits?",
        "[09:19] <alice> bob: ah it's a memory issue, thanks!",
        "[09:19] <eve> anyone free for code review?",
        "[09:20] <dave> charlie: no, rebase rewrites history but keeps your commits",
        "[09:20] <bob> eve: sure, post the PR link",
        "[09:21] <eve> bob: here you go github.com/project/pr/123"
    ],
    'tokenized': [
        ['09:15', 'alice', 'does', 'anyone', 'have', 'experience', 'with', 'docker', 'compose'],
        ['09:16', 'bob', 'alice', 'yeah', 'what', 's', 'the', 'issue'],
        ['09:16', 'charlie', 'morning', 'all', 'quick', 'question', 'about', 'git', 'rebase'],
        ['09:17', 'alice', 'bob', 'my', 'containers', 'keep', 'restarting', 'not', 'sure', 'why'],
        ['09:17', 'dave', 'charlie', 'shoot', 'what', 'do', 'you', 'need'],
        ['09:18', 'bob', 'alice', 'check', 'your', 'logs', 'with', 'docker', 'logs', 'container_name'],
        ['09:18', 'charlie', 'dave', 'if', 'i', 'rebase', 'on', 'main', 'will', 'i', 'lose', 'my', 'commits'],
        ['09:19', 'alice', 'bob', 'ah', 'it', 's', 'a', 'memory', 'issue', 'thanks'],
        ['09:19', 'eve', 'anyone', 'free', 'for', 'code', 'review'],
        ['09:20', 'dave', 'charlie', 'no', 'rebase', 'rewrites', 'history', 'but', 'keeps', 'your', 'commits'],
        ['09:20', 'bob', 'eve', 'sure', 'post', 'the', 'pr', 'link'],
        ['09:21', 'eve', 'bob', 'here', 'you', 'go', 'github', 'com', 'project', 'pr', '123']
    ],
    'date': ['2004-12-25', '2004-12-25', '2004-12-25', '2004-12-25', 
             '2004-12-25', '2004-12-25', '2004-12-25', '2004-12-25',
             '2004-12-25', '2004-12-25', '2004-12-25', '2004-12-25'],
    'connections': [
        [1],        # Node 1: alice starts docker question
        [1],        # Node 2: bob replies to alice (Node 1)
        [3],        # Node 3: charlie starts git question
        [2],        # Node 4: alice replies to bob (Node 2)
        [3],        # Node 5: dave replies to charlie (Node 3)
        [4],        # Node 6: bob replies to alice (Node 4)
        [5],        # Node 7: charlie replies to dave (Node 5)
        [6],        # Node 8: alice replies to bob (Node 6)
        [9],        # Node 9: eve starts code review question
        [7],        # Node 10: dave replies to charlie (Node 7)
        [9],        # Node 11: bob replies to eve (Node 9)
        [11]        # Node 12: eve replies to bob (Node 11)
    ]
}