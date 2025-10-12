from slang_dict import slang_dict, slang_set
from time_dict import months, days, hours
import pandas as pd # used for date conversion


def removeSlang(message): # expands slang 
  message_arr = message.split() 

  for index, word in enumerate(message_arr): 
    if word.lower() in slang_set: 
      message_arr[index] = slang_dict[word.lower()]

  message = " ".join(message_arr) # convert message back to string 
  return message

def format_message(subject_name=str, message=str, time_stamp=str): 
  # 2024-10-08 01:31:13 # original format of time_stamp
  # convert into : 
  # Sent from Me on Monday, October 18th 2024, 1:31 PM : I don't like the way you are talking to me.

  date = { 
      "year" : None, 
      "month" : None,
      "day" : None,
      "day_of_week" : None
  }

  time = { 
    "hour" : None, 
    "minute" : None, 
    "second" : None
  }

  parts = time_stamp.split(' ')

  d = pd.Timestamp(parts[0])
  day_of_week = d.day_name()

  raw_date = parts[0].split('-')
  # 2024, 10, 08
  date["year"] = raw_date[0] 
  date["month"] = months[raw_date[1]] # non-numerical conversion 
  date["day"] = raw_date[2] 
  date["day_of_week"] = day_of_week
  
  # 01, 31, 13
  raw_time = parts[1].split(':')
  time["hour"] = raw_time[0]
  time["minute"] = raw_time[1]
  time["second"] = raw_time[2] 

  time_format = sequence_time(date, time)
  if(subject_name == 'Me'): 
    sender_format = "Sent from Me on "
    formatted = sender_format + time_format + " : " + message + "\n"
  else: 
    sender_format = f"Sent from {subject_name} on "
    formatted = sender_format + time_format + " : " + message + "\n"
    
  return formatted

def sequence_time(date=dict, time=dict):
  hour = int(time["hour"])
  minute = time["minute"]

  # Convert to 12-hour format and determine AM/PM
  if hour == 0:
    hour_12 = 12
    period = "AM"
  elif hour < 12:
    hour_12 = hour
    period = "AM"
  elif hour == 12:
    hour_12 = 12
    period = "PM"
  else: # hour > 12
    hour_12 = hour - 12
    period = "PM"

  # Format: Monday, October 18 2024, 1:31 PM
  formatted = f"{date['day_of_week']}, {date['month']} {date['day']} {date['year']}, {hour_12}:{minute} {period}"

  return formatted


formatted = format_message("John", "Why were you angry last night?", "2024-10-08 01:31:13")
print(formatted)

def formatMyMessage(message, time_stamp): # format everytime I am the sender
  subject_name = 'Me'
  formatted_message = removeSlang(message)
  formatted = format_message(subject_name, formatted_message, time_stamp)
  return formatted 

def formatSenderMessage(subject_name, message, time_stamp): 
  expanded_slang = removeSlang(message)
  formatted_message = removeSlang(message)
  formatted_message = format_message(subject_name, formatted_message, time_stamp)
  return formatted_message


  