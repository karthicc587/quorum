"""Hand-labeled fixtures. Ground truth for the eval numbers in the writeup.

WAKE_CASES are the ways Whisper actually renders a name it has never seen:
dropped initialisms, phonetic guesses, run-together words. Collected from
small.en/medium.en behaviour on out-of-vocabulary proper nouns.
"""

WAKE_PHRASE = "AI Kartik"

# Severe mistranscriptions that fuzzy matching cannot reach without also
# firing on "artificial" and "karaoke". Documented misses, not bugs: a miss
# costs a repeated question, a false fire interrupts the meeting.
KNOWN_MISSES = ["A I Carte, do you have the numbers"]

# (transcript, should_wake)
WAKE_CASES = [
    # clean
    ("AI Kartik, are you working on the competitor analysis?", True),
    ("Hey AI Kartik, quick one for you.", True),
    ("A.I. Kartik, status?", True),
    # whisper phonetic drift
    ("Hey Cartik, are we on track?", True),
    ("AI Kartek, what's the status", True),
    ("ai kartik whats the update", True),
    ("Okay, AI Karthik — where are we", True),
    ("Hey, I Kartik, one question.", True),
    ("AIKartik can you confirm", True),
    # mid-sentence
    ("So before we move on, AI Kartik, is the draft ready?", True),
    ("Let's ask AI Kartik about the timeline.", True),
    # near misses that must NOT fire
    ("I think artificial intelligence is overrated.", False),
    ("Can you karaoke? Just kidding.", False),
    ("The AI tools we evaluated were expensive.", False),
    ("Let's talk about the article critique.", False),
    ("Kartik isn't here today, he sent the agent.", True),   # name alone, deliberate
    ("My car took a hit last week.", False),
    ("Anyway, I already emailed you about it.", False),
    ("Are the artifacts checked in?", False),
]

# (utterance, expected_tier, note)
ROUTER_CASES = [
    # --- answerable from the template KB ---
    ("are you working on the competitor analysis", "answer", "fact 1"),
    ("what's the status of the competitor analysis", "answer", "fact 1"),
    ("who is on the team", "answer", "fact 2"),
    ("did the last deliverable go in", "answer", "fact 3"),
    ("were there revisions requested on the last one", "answer", "fact 3"),
    ("how often does your team meet", "answer", "fact 4"),
    ("how long are the meetings", "answer", "fact 4"),
    ("how many competitors are in scope", "answer", "position 1"),
    ("when was the scope decided", "answer", "position 1"),
    ("are you paying for hosting", "answer", "position 2"),

    # --- commitment guard ---
    ("can you have it done by Monday", "escalate", "guard:commitment"),
    ("could you send me the draft tonight", "escalate", "guard:commitment"),
    ("will you take the write-up as well", "escalate", "guard:commitment"),
    ("what about end of week instead", "escalate", "guard:scheduling"),
    ("is Wednesday still realistic", "escalate", "guard:scheduling"),
    ("can you sign off on this", "escalate", "guard:commitment"),

    # --- opinion guard ---
    ("do you think we should pivot the business", "escalate", "guard:opinion"),
    ("what's your take on the second competitor", "escalate", "guard:opinion"),
    ("should we add a third competitor", "escalate", "guard:opinion"),
    ("would you rather present first or last", "escalate", "guard:opinion"),
    ("do you agree with that framing", "escalate", "guard:opinion"),

    # --- interpersonal guard ---
    ("is everyone else pulling their weight", "escalate", "guard:interpersonal"),
    ("whose fault was the delay", "escalate", "guard:interpersonal"),
    ("off the record, how's the team doing", "escalate", "guard:interpersonal"),

    # --- resource guard ---
    ("what's the budget for this", "escalate", "guard:resource"),
    ("did we pay for any of these tools", "escalate", "guard:resource"),

    # --- not in KB, no guard fires: the model must catch these ---
    ("what did the competitor analysis conclude", "escalate", "not in KB"),
    ("how many pages is the draft", "escalate", "not in KB"),
    ("which two competitors", "escalate", "not in KB — scope count only"),
    ("what grade did the last deliverable get", "escalate", "not in KB"),
    ("where are the files stored", "escalate", "not in KB"),
    ("what's the professor's name", "escalate", "not in KB"),
    ("has the third competitor been ruled out permanently", "escalate", "extends a position"),
    ("so the scope will stay at two for the final too", "escalate", "extends a position"),
    ("is the draft going to be any good", "escalate", "opinion + not in KB"),
    ("remind me what tool stack you settled on", "escalate", "vague, not literal"),
    ("did the meeting cadence change", "escalate", "not in KB — no change record"),
    ("is Kartik on the team", "answer", "fact 2"),
]
