message_chunks = [

# 1️⃣ Movie weekend plans
[
    # Section 1: Weekend plans discussion (3 bursts)
    # Burst 1: Initial invitation
    ('2024-10-15 14:30:00', '', '', 'Hey! Want to catch a movie this weekend?', '+15551234567'),
    ('2024-10-15 14:30:45', '', '', 'Heard the new sci-fi one is really good', '+15551234567'),
    ('2024-10-15 14:31:20', '', '', 'Reviews are amazing', '+15551234567'),

    # Burst 2: Response (8 min gap)
    ('2024-10-15 14:39:00', 'me', '', 'Oh yeah! I saw the trailer', '+19365539666'),
    ('2024-10-15 14:39:30', 'me', '', 'Looks incredible', '+19365539666'),

    # Burst 3: Time coordination (12 min gap)
    ('2024-10-15 14:51:00', '', '', 'Saturday evening work for you?', '+15551234567'),
    ('2024-10-15 14:52:00', 'me', '', 'Perfect! 7pm showing?', '+19365539666'),
    ('2024-10-15 14:52:30', '', '', 'Sounds great', '+15551234567'),

    # Gap > 30 minutes (conversation break)
    # Section 2: Day-of coordination
    # Burst 1: Running late
    ('2024-10-16 18:45:00', 'me', '', 'Running a bit behind', '+19365539666'),
    ('2024-10-16 18:45:30', 'me', '', 'Parking was a nightmare', '+19365539666'),
    ('2024-10-16 18:46:00', 'me', '', 'Be there in 5', '+19365539666'),

    # Burst 2: Arrival (10 min gap)
    ('2024-10-16 18:56:00', 'me', '', "I'm at the ticket counter", '+19365539666'),
    ('2024-10-16 18:56:30', '', '', 'See you! Grabbing popcorn', '+15551234567'),

    # Gap > 30 minutes (movie ended)
    # Section 3: Post-movie discussion
    # Burst 1: Initial reaction
    ('2024-10-16 21:30:00', '', '', 'WOW that ending!', '+15551234567'),
    ('2024-10-16 21:30:15', '', '', 'Did NOT see that coming', '+15551234567'),
    ('2024-10-16 21:30:45', 'me', '', 'Right?? Totally unexpected', '+19365539666'),
    ('2024-10-16 21:31:00', 'me', '', 'The visuals were insane too', '+19365539666'),

    # Burst 2: Making plans again (15 min gap)
    ('2024-10-16 21:46:00', '', '', 'We should do this more often', '+15551234567'),
    ('2024-10-16 21:46:30', 'me', '', 'Definitely! Next weekend?', '+19365539666'),
    ('2024-10-16 21:47:00', '', '', "I'm in!", '+15551234567'),

    # Gap > 30 minutes (late night)
    # Section 4: Next day follow-up
    ('2024-10-17 10:15:00', 'me', '', 'Still thinking about that movie', '+19365539666'),
    ('2024-10-17 10:15:30', 'me', '', 'Might see it again tbh', '+19365539666'),
    ('2024-10-17 10:16:15', '', '', 'Haha same! It was that good', '+15551234567'),
],

# 2️⃣ Group project coordination
[
    # Section 1: Planning the meeting (2 bursts)
    # Burst 1: Scheduling discussion
    ('2024-11-01 09:15:00', '', '', 'Should we meet this afternoon to work on the report?', '+18015550123'),
    ('2024-11-01 09:15:45', '', '', 'The professor extended the deadline btw', '+18015550123'),
    ('2024-11-01 09:16:10', 'me', '', 'Nice! 3pm at the library?', '+19365539666'),

    # Burst 2: Confirmation (7 min gap)
    ('2024-11-01 09:23:00', '', '', '3pm works! Bring your laptop so we can merge sections', '+18015550123'),
    ('2024-11-01 09:24:00', 'me', '', 'I’ll print the rubric before coming', '+19365539666'),

    # Gap > 30 minutes (before meetup)
    # Section 2: Study session
    # Burst 1: Arriving at location
    ('2024-11-01 15:02:00', 'me', '', 'I’m in the study room, table 5', '+19365539666'),
    ('2024-11-01 15:02:30', '', '', 'On my way, grabbing coffee first', '+18015550123'),

    # Burst 2: Progress updates (2 hr gap)
    ('2024-11-01 17:35:00', 'me', '', 'Okay, we’re like 90% done!', '+19365539666'),
    ('2024-11-01 17:36:10', '', '', 'Let’s review sources tomorrow morning?', '+18015550123'),

    # Gap > 12 hours (overnight)
    # Section 3: Final submission
    # Burst 1: Morning edits
    ('2024-11-02 09:00:00', '', '', 'Morning! Reviewing now. We missed one citation', '+18015550123'),
    ('2024-11-02 09:01:00', 'me', '', 'Got it, adding it now', '+19365539666'),
    ('2024-11-02 09:02:00', '', '', 'Final version done?', '+18015550123'),
    ('2024-11-02 09:02:30', 'me', '', 'Yup, submitting it!', '+19365539666'),

    # Burst 2: Completion confirmation (20 min gap)
    ('2024-11-02 10:20:00', 'me', '', 'Submitted ✅', '+19365539666'),
    ('2024-11-02 10:21:00', '', '', 'Great job team', '+18015550123'),
],

# 3️⃣ Travel coordination
[
    # Section 1: Trip prep (2 bursts)
    # Burst 1: Booking discussion
    ('2024-12-05 08:00:00', '', '', 'Booked my flight to Denver! 10am Saturday', '+14085550111'),
    ('2024-12-05 08:00:40', '', '', 'You find a hotel yet?', '+14085550111'),
    ('2024-12-05 08:02:00', 'me', '', 'Yeah, got one downtown near Union Station', '+19365539666'),

    # Burst 2: Reaction (1 min gap)
    ('2024-12-05 08:02:30', '', '', 'Sweet, we can walk to most places then', '+14085550111'),

    # Gap > 24 hours (trip approaching)
    # Section 2: Packing and departure
    # Burst 1: Night before
    ('2024-12-06 17:00:00', 'me', '', 'Packing rn. You bringing hiking gear?', '+19365539666'),
    ('2024-12-06 17:01:00', '', '', 'Yup! Weather looks great', '+14085550111'),

    # Burst 2: Day of flight (16 hr gap)
    ('2024-12-07 09:00:00', 'me', '', 'Boarding soon', '+19365539666'),
    ('2024-12-07 09:01:00', '', '', 'Same, see you at baggage claim?', '+14085550111'),
    ('2024-12-07 12:10:00', '', '', 'Landed!', '+14085550111'),
    ('2024-12-07 12:11:00', 'me', '', 'Cool, Ubering now', '+19365539666'),

    # Gap > 8 hours (evening plans)
    # Section 3: Dinner plans
    # Burst 1: Restaurant choice
    ('2024-12-07 20:00:00', '', '', 'Dinner spot tonight?', '+14085550111'),
    ('2024-12-07 20:01:00', 'me', '', 'Found a tapas place nearby', '+19365539666'),
    ('2024-12-07 20:02:30', '', '', 'Perfect, 7:30 reservation?', '+14085550111'),
    ('2024-12-07 20:03:00', 'me', '', 'Booked ✅', '+19365539666'),

    # Burst 2: Post-dinner reflection (13 hr gap)
    ('2024-12-08 09:15:00', '', '', 'That was so good last night', '+14085550111'),
    ('2024-12-08 09:16:30', 'me', '', 'Yeah, food and view were top tier', '+19365539666'),
],

# 4️⃣ Work deadline crunch
[
    # Section 1: Debugging session (2 bursts)
    # Burst 1: Issue discovery
    ('2025-01-12 13:00:00', '', '', 'Did you finish the analytics script?', '+13105550122'),
    ('2025-01-12 13:01:10', 'me', '', 'Almost, running into a JSON parsing issue', '+19365539666'),
    ('2025-01-12 13:02:00', '', '', 'Need help? I can check the logs', '+13105550122'),
    ('2025-01-12 13:02:45', 'me', '', 'Sure, sent them in Slack', '+19365539666'),

    # Burst 2: Fix confirmation (3 min gap)
    ('2025-01-12 13:05:30', '', '', 'Ah yeah, missing a comma on line 242', '+13105550122'),
    ('2025-01-12 13:06:15', 'me', '', 'Fixed it! Thanks man', '+19365539666'),

    # Gap > 1 hr (later testing)
    # Section 2: Deployment
    ('2025-01-12 14:30:00', 'me', '', 'Script passed validation ✅', '+19365539666'),
    ('2025-01-12 14:31:00', '', '', 'Let’s deploy then', '+13105550122'),

    # Burst 2: Post-deployment (90 min gap)
    ('2025-01-12 16:00:00', 'me', '', 'Deployed to staging', '+19365539666'),
    ('2025-01-12 16:01:00', '', '', 'Cool, will review in 10', '+13105550122'),

    # Gap > 12 hours
    # Section 3: After feedback
    ('2025-01-13 09:00:00', '', '', 'Boss said great job btw', '+13105550122'),
    ('2025-01-13 09:01:15', 'me', '', 'Nice! Appreciate it', '+19365539666'),
],
]



