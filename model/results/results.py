def aggregate_results(): # data will be a dictionary with multiple fields 
  with open("results.json", "r") as file: 
    lines = file.readlines() 
  
  return lines