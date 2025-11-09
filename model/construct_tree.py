def build_decision_tree(messages, connections): 
  print("HI")


example = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8],  # List of all message IDs for this day
    
    'raw': [
        "[14:32] <BurgerMann> hey how do I install nvidia drivers?",
        "[14:33] <delire> BurgerMann: sudo apt-get install nvidia-drivers",
        "[14:33] <Seveas> anyone know if mysql is down?",
        "[14:34] <BurgerMann> delire: which version should I use?",
        "[14:34] <delire> Seveas: works for me",
        "[14:35] <techguy> BurgerMann: try nvidia-driver-470",
        "[14:35] <Seveas> delire: hmm weird, must be my connection",
        "[14:36] <BurgerMann> thanks both!"
    ],
    
    'ascii': [ /* same as raw, ASCII-converted */ ],
    
    'tokenized': [ /* tokenized versions of each message */ ],
    
    'date': ['2004-12-25', '2004-12-25', '2004-12-25', '2004-12-25', 
             '2004-12-25', '2004-12-25', '2004-12-25', '2004-12-25'],
    
    'connections': [
        [1],        # Node 1: links to itself (starts new conversation)
        [1],        # Node 2: replies to Node 1
        [3],        # Node 3: links to itself (starts new conversation)
        [2],        # Node 4: replies to Node 2
        [3],        # Node 5: replies to Node 3
        [1, 2],     # Node 6: replies to both Node 1 and Node 2
        [5],        # Node 7: replies to Node 5
        [2, 6]      # Node 8: replies to Node 2 and Node 6
    ]
}