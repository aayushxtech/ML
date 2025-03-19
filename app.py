# app.py
from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
from statistics import mean
import csv
import os

app = Flask(__name__)

# Load batter data from CSV
def load_batters_from_csv():
    batters = []
    csv_path = os.path.join(os.path.dirname(__file__), 'unified_batter_stats.csv')
    try:
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if row:  # Ensure row is not empty
                    batters.append(row[0])  # First column contains batter name
    except Exception as e:
        print(f"Error loading batters from CSV: {e}")
        # Fallback to some default batters if CSV loading fails
        batters = ["Virat Kohli", "MS Dhoni", "Rohit Sharma", "KL Rahul"]
    
    return sorted(batters)  # Return sorted list of batters

# Load the trained model
model_filename = "trained_model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Define IPL teams
ipl_teams = ["CSK", "MI", "RCB", "KKR", "DC", "SRH", "PBKS", "RR", "GT", "LSG"]

# Define IPL venues with realistic difficulty ratings (based on batting averages)
ipl_venues = {
    "MA Chidambaram Stadium": {"venue_difficulty": 27.4},  # Chennai - spin friendly
    "Wankhede Stadium": {"venue_difficulty": 31.2},        # Mumbai - batting friendly
    "M Chinnaswamy Stadium": {"venue_difficulty": 33.8},   # Bangalore - highest scoring
    "Eden Gardens": {"venue_difficulty": 29.5},            # Kolkata - balanced
    "Arun Jaitley Stadium": {"venue_difficulty": 30.7},    # Delhi - batting friendly
    "Rajiv Gandhi Stadium": {"venue_difficulty": 28.3},    # Hyderabad - balanced
    "Punjab Cricket Association Stadium": {"venue_difficulty": 31.5},  # Mohali - batting friendly
    "Sawai Mansingh Stadium": {"venue_difficulty": 29.1},  # Jaipur - balanced
    "Narendra Modi Stadium": {"venue_difficulty": 28.4},   # Ahmedabad - large ground
    "Ekana Cricket Stadium": {"venue_difficulty": 26.8}    # Lucknow - slow pitch
}

# Define batter profiles with realistic values
batter_profiles = {
    "Virat Kohli": {
        "average_runs": 37.8,
        "average_strike_rate": 135.7,
        "total_balls_faced": 1842,
        "total_dismissals": 47,
        "team_averages": {
            "CSK": 36.4, "MI": 39.2, "RCB": 0.0, "KKR": 41.3, "DC": 38.7,
            "SRH": 35.9, "PBKS": 42.2, "RR": 40.8, "GT": 33.5, "LSG": 35.2
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 33.6, "Wankhede Stadium": 40.2, "M Chinnaswamy Stadium": 48.7,
            "Eden Gardens": 42.9, "Arun Jaitley Stadium": 43.5, "Rajiv Gandhi Stadium": 35.8,
            "Punjab Cricket Association Stadium": 41.3, "Sawai Mansingh Stadium": 39.5,
            "Narendra Modi Stadium": 34.8, "Ekana Cricket Stadium": 32.4
        }
    },
    "Rohit Sharma": {
        "average_runs": 29.4,
        "average_strike_rate": 133.8,
        "total_balls_faced": 1738,
        "total_dismissals": 58,
        "team_averages": {
            "CSK": 33.6, "MI": 0.0, "RCB": 32.8, "KKR": 31.4, "DC": 34.5,
            "SRH": 27.8, "PBKS": 31.2, "RR": 29.7, "GT": 28.5, "LSG": 30.1
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 26.4, "Wankhede Stadium": 38.7, "M Chinnaswamy Stadium": 34.9,
            "Eden Gardens": 33.5, "Arun Jaitley Stadium": 32.8, "Rajiv Gandhi Stadium": 25.6,
            "Punjab Cricket Association Stadium": 30.2, "Sawai Mansingh Stadium": 28.3,
            "Narendra Modi Stadium": 31.4, "Ekana Cricket Stadium": 25.2
        }
    },
    "MS Dhoni": {
        "average_runs": 24.8,
        "average_strike_rate": 153.5,
        "total_balls_faced": 982,
        "total_dismissals": 38,
        "team_averages": {
            "CSK": 0.0, "MI": 22.7, "RCB": 26.3, "KKR": 25.2, "DC": 23.8,
            "SRH": 21.9, "PBKS": 27.1, "RR": 24.5, "GT": 20.2, "LSG": 22.4
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 33.6, "Wankhede Stadium": 25.7, "M Chinnaswamy Stadium": 27.2,
            "Eden Gardens": 30.1, "Arun Jaitley Stadium": 23.8, "Rajiv Gandhi Stadium": 22.4,
            "Punjab Cricket Association Stadium": 24.9, "Sawai Mansingh Stadium": 21.3,
            "Narendra Modi Stadium": 19.8, "Ekana Cricket Stadium": 21.5
        }
    },
    "KL Rahul": {
        "average_runs": 43.2,
        "average_strike_rate": 138.4,
        "total_balls_faced": 1565,
        "total_dismissals": 36,
        "team_averages": {
            "CSK": 40.4, "MI": 45.2, "RCB": 46.8, "KKR": 41.7, "DC": 44.2,
            "SRH": 42.3, "PBKS": 0.0, "RR": 43.9, "GT": 39.5, "LSG": 0.0
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 37.2, "Wankhede Stadium": 44.5, "M Chinnaswamy Stadium": 49.6,
            "Eden Gardens": 40.8, "Arun Jaitley Stadium": 42.3, "Rajiv Gandhi Stadium": 39.7,
            "Punjab Cricket Association Stadium": 52.4, "Sawai Mansingh Stadium": 43.2,
            "Narendra Modi Stadium": 38.6, "Ekana Cricket Stadium": 46.8
        }
    },
    "Jos Buttler": {
        "average_runs": 41.8,
        "average_strike_rate": 148.9,
        "total_balls_faced": 1428,
        "total_dismissals": 34,
        "team_averages": {
            "CSK": 38.5, "MI": 0.0, "RCB": 43.7, "KKR": 39.6, "DC": 42.3,
            "SRH": 37.2, "PBKS": 42.9, "RR": 0.0, "GT": 36.4, "LSG": 39.1
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 32.5, "Wankhede Stadium": 42.6, "M Chinnaswamy Stadium": 46.2,
            "Eden Gardens": 39.4, "Arun Jaitley Stadium": 40.7, "Rajiv Gandhi Stadium": 35.8,
            "Punjab Cricket Association Stadium": 40.2, "Sawai Mansingh Stadium": 49.4,
            "Narendra Modi Stadium": 36.3, "Ekana Cricket Stadium": 33.9
        }
    },
    "Shubman Gill": {
        "average_runs": 39.7,
        "average_strike_rate": 139.5,
        "total_balls_faced": 1376,
        "total_dismissals": 34,
        "team_averages": {
            "CSK": 36.2, "MI": 40.5, "RCB": 38.4, "KKR": 0.0, "DC": 37.3,
            "SRH": 35.1, "PBKS": 38.7, "RR": 37.2, "GT": 0.0, "LSG": 34.5
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 33.8, "Wankhede Stadium": 40.6, "M Chinnaswamy Stadium": 42.9,
            "Eden Gardens": 45.3, "Arun Jaitley Stadium": 37.4, "Rajiv Gandhi Stadium": 35.7,
            "Punjab Cricket Association Stadium": 38.2, "Sawai Mansingh Stadium": 35.6,
            "Narendra Modi Stadium": 44.8, "Ekana Cricket Stadium": 33.1
        }
    },
    "Rishabh Pant": {
        "average_runs": 35.6,
        "average_strike_rate": 152.7,
        "total_balls_faced": 1247,
        "total_dismissals": 35,
        "team_averages": {
            "CSK": 33.2, "MI": 37.4, "RCB": 38.1, "KKR": 35.9, "DC": 0.0,
            "SRH": 32.7, "PBKS": 38.3, "RR": 36.8, "GT": 33.4, "LSG": 35.2
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 30.5, "Wankhede Stadium": 37.9, "M Chinnaswamy Stadium": 42.3,
            "Eden Gardens": 36.2, "Arun Jaitley Stadium": 45.6, "Rajiv Gandhi Stadium": 32.4,
            "Punjab Cricket Association Stadium": 36.8, "Sawai Mansingh Stadium": 34.7,
            "Narendra Modi Stadium": 31.6, "Ekana Cricket Stadium": 29.8
        }
    },
    "Suryakumar Yadav": {
        "average_runs": 34.2,
        "average_strike_rate": 155.8,
        "total_balls_faced": 1189,
        "total_dismissals": 35,
        "team_averages": {
            "CSK": 35.7, "MI": 0.0, "RCB": 37.4, "KKR": 0.0, "DC": 33.9,
            "SRH": 31.8, "PBKS": 36.2, "RR": 35.1, "GT": 32.3, "LSG": 34.7
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 29.6, "Wankhede Stadium": 46.8, "M Chinnaswamy Stadium": 40.5,
            "Eden Gardens": 37.4, "Arun Jaitley Stadium": 32.8, "Rajiv Gandhi Stadium": 30.5,
            "Punjab Cricket Association Stadium": 34.2, "Sawai Mansingh Stadium": 33.7,
            "Narendra Modi Stadium": 31.2, "Ekana Cricket Stadium": 28.9
        }
    },
    "Sanju Samson": {
        "average_runs": 32.7,
        "average_strike_rate": 147.3,
        "total_balls_faced": 1285,
        "total_dismissals": 39,
        "team_averages": {
            "CSK": 30.4, "MI": 33.8, "RCB": 35.2, "KKR": 31.9, "DC": 32.5,
            "SRH": 29.7, "PBKS": 34.3, "RR": 0.0, "GT": 29.1, "LSG": 31.8
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 28.3, "Wankhede Stadium": 34.6, "M Chinnaswamy Stadium": 38.9,
            "Eden Gardens": 33.2, "Arun Jaitley Stadium": 31.7, "Rajiv Gandhi Stadium": 29.5,
            "Punjab Cricket Association Stadium": 33.9, "Sawai Mansingh Stadium": 42.7,
            "Narendra Modi Stadium": 27.8, "Ekana Cricket Stadium": 26.4
        }
    },
    "Yashasvi Jaiswal": {
        "average_runs": 37.9,
        "average_strike_rate": 148.2,
        "total_balls_faced": 1062,
        "total_dismissals": 28,
        "team_averages": {
            "CSK": 35.2, "MI": 39.1, "RCB": 40.3, "KKR": 37.8, "DC": 38.5,
            "SRH": 36.4, "PBKS": 39.6, "RR": 0.0, "GT": 34.7, "LSG": 36.8
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 32.6, "Wankhede Stadium": 39.8, "M Chinnaswamy Stadium": 43.2,
            "Eden Gardens": 38.5, "Arun Jaitley Stadium": 37.9, "Rajiv Gandhi Stadium": 35.6,
            "Punjab Cricket Association Stadium": 38.7, "Sawai Mansingh Stadium": 46.8,
            "Narendra Modi Stadium": 34.2, "Ekana Cricket Stadium": 31.9
        }
    },
    "Shreyas Iyer": {
        "average_runs": 33.4,
        "average_strike_rate": 136.8,
        "total_balls_faced": 1154,
        "total_dismissals": 34,
        "team_averages": {
            "CSK": 31.5, "MI": 34.7, "RCB": 35.2, "KKR": 0.0, "DC": 0.0,
            "SRH": 30.8, "PBKS": 34.9, "RR": 32.7, "GT": 30.2, "LSG": 32.3
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 29.3, "Wankhede Stadium": 38.6, "M Chinnaswamy Stadium": 36.4,
            "Eden Gardens": 42.7, "Arun Jaitley Stadium": 40.9, "Rajiv Gandhi Stadium": 31.2,
            "Punjab Cricket Association Stadium": 34.3, "Sawai Mansingh Stadium": 32.1,
            "Narendra Modi Stadium": 29.8, "Ekana Cricket Stadium": 27.9
        }
    },
    "David Warner": {
        "average_runs": 33.1,
        "average_strike_rate": 100.9,
        "total_balls_faced": 1590,
        "total_dismissals": 42,
        "team_averages": {
            "CSK": 33.1, "MI": 33.0, "RCB": 39.1, "KKR": 39.0, "DC": 26.0,
            "SRH": 20.3, "PBKS": 47.7, "RR": 36.4, "GT": 19.5, "LSG": 17.8
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 29.5, "Wankhede Stadium": 33.0, "M Chinnaswamy Stadium": 39.1,
            "Eden Gardens": 38.0, "Arun Jaitley Stadium": 33.5, "Rajiv Gandhi Stadium": 31.5,
            "Punjab Cricket Association Stadium": 36.0, "Sawai Mansingh Stadium": 34.0,
            "Narendra Modi Stadium": 24.5, "Ekana Cricket Stadium": 20.0
        }
    },
    "Faf du Plessis": {
        "average_runs": 39.4,
        "average_strike_rate": 145.3,
        "total_balls_faced": 1320,
        "total_dismissals": 32,
        "team_averages": {
            "CSK": 40.5, "MI": 28.1, "RCB": 29.5, "KKR": 27.1, "DC": 25.6,
            "SRH": 31.7, "PBKS": 50.2, "RR": 32.4, "GT": 32.0, "LSG": 47.6
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 39.4, "Wankhede Stadium": 28.5, "M Chinnaswamy Stadium": 29.5,
            "Eden Gardens": 27.0, "Arun Jaitley Stadium": 28.0, "Rajiv Gandhi Stadium": 32.0,
            "Punjab Cricket Association Stadium": 35.0, "Sawai Mansingh Stadium": 30.0,
            "Narendra Modi Stadium": 32.0, "Ekana Cricket Stadium": 30.0
        }
    },
    "AB de Villiers": {
        "average_runs": 21.3,
        "average_strike_rate": 107.0,
        "total_balls_faced": 1100,
        "total_dismissals": 52,
        "team_averages": {
            "CSK": 21.3, "MI": 33.0, "RCB": 28.8, "KKR": 24.9, "DC": 29.8,
            "SRH": 31.8, "PBKS": 33.8, "RR": 32.6, "GT": 0.0, "LSG": 0.0
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 21.3, "Wankhede Stadium": 33.0, "M Chinnaswamy Stadium": 28.8,
            "Eden Gardens": 26.0, "Arun Jaitley Stadium": 32.0, "Rajiv Gandhi Stadium": 31.8,
            "Punjab Cricket Association Stadium": 33.0, "Sawai Mansingh Stadium": 30.0,
            "Narendra Modi Stadium": 0.0, "Ekana Cricket Stadium": 0.0
        }
    },
    "Ruturaj Gaikwad": {
        "average_runs": 37.1,
        "average_strike_rate": 95.5,
        "total_balls_faced": 1250,
        "total_dismissals": 30,
        "team_averages": {
            "CSK": 0.0, "MI": 29.8, "RCB": 30.7, "KKR": 40.9, "DC": 29.8,
            "SRH": 56.3, "PBKS": 62.0, "RR": 30.0, "GT": 50.0, "LSG": 45.8
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 35.0, "Wankhede Stadium": 30.0, "M Chinnaswamy Stadium": 31.0,
            "Eden Gardens": 41.0, "Arun Jaitley Stadium": 30.0, "Rajiv Gandhi Stadium": 35.0,
            "Punjab Cricket Association Stadium": 45.0, "Sawai Mansingh Stadium": 33.0,
            "Narendra Modi Stadium": 42.0, "Ekana Cricket Stadium": 35.0
        }
    },
    "Kane Williamson": {
        "average_runs": 34.8,
        "average_strike_rate": 128.3,
        "total_balls_faced": 1150,
        "total_dismissals": 33,
        "team_averages": {
            "CSK": 34.8, "MI": 8.5, "RCB": 40.8, "KKR": 22.6, "DC": 30.3,
            "SRH": 0.0, "PBKS": 28.4, "RR": 27.6, "GT": 31.0, "LSG": 8.5
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 34.8, "Wankhede Stadium": 18.0, "M Chinnaswamy Stadium": 40.8,
            "Eden Gardens": 25.0, "Arun Jaitley Stadium": 32.0, "Rajiv Gandhi Stadium": 35.0,
            "Punjab Cricket Association Stadium": 28.0, "Sawai Mansingh Stadium": 30.0,
            "Narendra Modi Stadium": 31.0, "Ekana Cricket Stadium": 12.0
        }
    },
    "Quinton de Kock": {
        "average_runs": 25.9,
        "average_strike_rate": 112.0,
        "total_balls_faced": 1300,
        "total_dismissals": 45,
        "team_averages": {
            "CSK": 25.9, "MI": 17.6, "RCB": 33.7, "KKR": 30.3, "DC": 29.7,
            "SRH": 31.0, "PBKS": 37.0, "RR": 33.7, "GT": 23.5, "LSG": 0.0
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 25.9, "Wankhede Stadium": 24.0, "M Chinnaswamy Stadium": 33.7,
            "Eden Gardens": 30.0, "Arun Jaitley Stadium": 28.0, "Rajiv Gandhi Stadium": 31.0,
            "Punjab Cricket Association Stadium": 37.0, "Sawai Mansingh Stadium": 33.0,
            "Narendra Modi Stadium": 23.5, "Ekana Cricket Stadium": 25.0
        }
    },
    "Nicholas Pooran": {
        "average_runs": 24.9,
        "average_strike_rate": 137.9,
        "total_balls_faced": 950,
        "total_dismissals": 38,
        "team_averages": {
            "CSK": 24.9, "MI": 25.6, "RCB": 20.3, "KKR": 21.7, "DC": 31.6,
            "SRH": 34.4, "PBKS": 0.0, "RR": 20.0, "GT": 14.6, "LSG": 34.0
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 24.9, "Wankhede Stadium": 25.6, "M Chinnaswamy Stadium": 20.3,
            "Eden Gardens": 21.7, "Arun Jaitley Stadium": 31.6, "Rajiv Gandhi Stadium": 34.4,
            "Punjab Cricket Association Stadium": 28.0, "Sawai Mansingh Stadium": 20.0,
            "Narendra Modi Stadium": 14.6, "Ekana Cricket Stadium": 34.0
        }
    },
    "Glenn Maxwell": {
        "average_runs": 27.1,
        "average_strike_rate": 149.9,
        "total_balls_faced": 1100,
        "total_dismissals": 40,
        "team_averages": {
            "CSK": 27.1, "MI": 24.3, "RCB": 11.4, "KKR": 24.0, "DC": 26.9,
            "SRH": 22.6, "PBKS": 6.0, "RR": 24.5, "GT": 22.0, "LSG": 19.0
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 27.1, "Wankhede Stadium": 24.3, "M Chinnaswamy Stadium": 11.4,
            "Eden Gardens": 24.0, "Arun Jaitley Stadium": 26.9, "Rajiv Gandhi Stadium": 22.6,
            "Punjab Cricket Association Stadium": 14.0, "Sawai Mansingh Stadium": 24.5,
            "Narendra Modi Stadium": 22.0, "Ekana Cricket Stadium": 19.0
        }
    },
    "Hardik Pandya": {
        "average_runs": 13.1,
        "average_strike_rate": 133.9,
        "total_balls_faced": 980,
        "total_dismissals": 75,
        "team_averages": {
            "CSK": 13.1, "MI": 17.3, "RCB": 19.9, "KKR": 28.2, "DC": 22.4,
            "SRH": 14.6, "PBKS": 18.9, "RR": 32.7, "GT": 0.0, "LSG": 25.2
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 13.1, "Wankhede Stadium": 17.3, "M Chinnaswamy Stadium": 19.9,
            "Eden Gardens": 28.2, "Arun Jaitley Stadium": 22.4, "Rajiv Gandhi Stadium": 14.6,
            "Punjab Cricket Association Stadium": 18.9, "Sawai Mansingh Stadium": 32.7,
            "Narendra Modi Stadium": 20.0, "Ekana Cricket Stadium": 25.2
        }
    },
    "Ravindra Jadeja": {
        "average_runs": 19.8,
        "average_strike_rate": 125.1,
        "total_balls_faced": 1050,
        "total_dismissals": 53,
        "team_averages": {
            "CSK": 0.0, "MI": 13.5, "RCB": 12.7, "KKR": 17.6, "DC": 19.9,
            "SRH": 15.1, "PBKS": 18.3, "RR": 15.3, "GT": 14.2, "LSG": 23.3
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 19.8, "Wankhede Stadium": 15.0, "M Chinnaswamy Stadium": 14.0,
            "Eden Gardens": 18.0, "Arun Jaitley Stadium": 20.0, "Rajiv Gandhi Stadium": 15.1,
            "Punjab Cricket Association Stadium": 18.3, "Sawai Mansingh Stadium": 15.3,
            "Narendra Modi Stadium": 14.2, "Ekana Cricket Stadium": 23.3
        }
    },
    "Andre Russell": {
        "average_runs": 24.7,
        "average_strike_rate": 135.0,
        "total_balls_faced": 870,
        "total_dismissals": 35,
        "team_averages": {
            "CSK": 24.7, "MI": 17.0, "RCB": 28.3, "KKR": 0.0, "DC": 34.0,
            "SRH": 20.3, "PBKS": 31.4, "RR": 16.7, "GT": 27.7, "LSG": 17.3
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 24.7, "Wankhede Stadium": 17.0, "M Chinnaswamy Stadium": 28.3,
            "Eden Gardens": 30.0, "Arun Jaitley Stadium": 34.0, "Rajiv Gandhi Stadium": 20.3,
            "Punjab Cricket Association Stadium": 31.4, "Sawai Mansingh Stadium": 16.7,
            "Narendra Modi Stadium": 27.7, "Ekana Cricket Stadium": 17.3
        }
    },
    "Devon Conway": {
        "average_runs": 61.3,
        "average_strike_rate": 133.8,
        "total_balls_faced": 920,
        "total_dismissals": 15,
        "team_averages": {
            "CSK": 0.0, "MI": 14.7, "RCB": 69.5, "KKR": 29.7, "DC": 61.3,
            "SRH": 81.0, "PBKS": 92.0, "RR": 24.7, "GT": 23.3, "LSG": 47.0
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 48.0, "Wankhede Stadium": 18.0, "M Chinnaswamy Stadium": 69.5,
            "Eden Gardens": 30.0, "Arun Jaitley Stadium": 61.3, "Rajiv Gandhi Stadium": 81.0,
            "Punjab Cricket Association Stadium": 92.0, "Sawai Mansingh Stadium": 25.0,
            "Narendra Modi Stadium": 23.3, "Ekana Cricket Stadium": 47.0
        }
    },
    "Rinku Singh": {
        "average_runs": 24.2,
        "average_strike_rate": 94.0,
        "total_balls_faced": 880,
        "total_dismissals": 32,
        "team_averages": {
            "CSK": 24.2, "MI": 13.2, "RCB": 23.3, "KKR": 0.0, "DC": 16.5,
            "SRH": 32.4, "PBKS": 10.0, "RR": 20.3, "GT": 34.0, "LSG": 32.3
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 24.2, "Wankhede Stadium": 15.0, "M Chinnaswamy Stadium": 23.3,
            "Eden Gardens": 28.0, "Arun Jaitley Stadium": 16.5, "Rajiv Gandhi Stadium": 32.4,
            "Punjab Cricket Association Stadium": 10.0, "Sawai Mansingh Stadium": 20.3,
            "Narendra Modi Stadium": 34.0, "Ekana Cricket Stadium": 32.3
        }
    },
    "Tilak Varma": {
        "average_runs": 34.5,
        "average_strike_rate": 120.2,
        "total_balls_faced": 900,
        "total_dismissals": 26,
        "team_averages": {
            "CSK": 34.5, "MI": 0.0, "RCB": 42.0, "KKR": 22.0, "DC": 30.6,
            "SRH": 36.5, "PBKS": 24.8, "RR": 44.4, "GT": 22.8, "LSG": 24.3
        },
        "venue_averages": {
            "MA Chidambaram Stadium": 34.5, "Wankhede Stadium": 36.0, "M Chinnaswamy Stadium": 42.0,
            "Eden Gardens": 22.0, "Arun Jaitley Stadium": 30.6, "Rajiv Gandhi Stadium": 36.5,
            "Punjab Cricket Association Stadium": 24.8, "Sawai Mansingh Stadium": 44.4,
            "Narendra Modi Stadium": 22.8, "Ekana Cricket Stadium": 24.3
        }
    }
}

# Calculate team bowling strengths based on opposition performance against them
def calculate_bowling_strengths():
    team_bowling_strengths = {}
    
    for team in ipl_teams:
        # Gather all batting averages against this team
        batting_averages_against = []
        for batter_name, batter_data in batter_profiles.items():
            # Skip if batter plays for this team (0.0 in data)
            if batter_data["team_averages"][team] > 0:
                batting_averages_against.append(batter_data["team_averages"][team])
        
        # Lower average = better bowling, so inverse the relationship
        # Scale to reasonable bowling strength values (7.5-9.5)
        if batting_averages_against:
            avg_batting_against = mean(batting_averages_against)
            # Lower batting average -> higher bowling strength
            bowling_strength = 16.5 - (avg_batting_against / 5)  
            # Ensure within reasonable range
            bowling_strength = max(min(bowling_strength, 9.5), 7.5)
        else:
            bowling_strength = 8.5  # Default value if no data
        
        team_bowling_strengths[team] = round(bowling_strength, 1)
    
    return team_bowling_strengths

# Calculate team batting strengths based on performance
def calculate_batting_strengths():
    team_batting_strengths = {}
    
    # For each team, find the batters in our profile and calculate average
    for team in ipl_teams:
        team_batters = []
        for batter_name, batter_data in batter_profiles.items():
            # Check if batter plays for this team (0.0 in data for their own team)
            batters_teams = [t for t, avg in batter_data["team_averages"].items() if avg == 0.0]
            if team in batters_teams:
                team_batters.append(batter_data["average_runs"])
        
        # Calculate average batting strength for team
        if team_batters:
            team_batting_strengths[team] = mean(team_batters)
        else:
            team_batting_strengths[team] = 35.0  # Default if no data
    
    return team_batting_strengths

# Calculate dynamically based on player data
team_bowling_strengths = calculate_bowling_strengths()
team_batting_strengths = calculate_batting_strengths()

@app.route("/", methods=["GET"])
def index():
    """Serve the frontend HTML file"""
    return send_from_directory(os.path.dirname(__file__), 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract parameters from request
        batter_name = data.get("batter_name", "")
        opponent_team = data.get("opponent_team", "")
        venue_name = data.get("venue_name", "")
        
        # Validate inputs
        if not batter_name:
            return jsonify({"error": "Batter name is required"}), 400
            
        # Check if batter exists
        if batter_name not in batter_profiles:
            return jsonify({"error": f"Batter {batter_name} not found in database"})
            
        # Check if venue exists
        if venue_name not in ipl_venues:
            return jsonify({"error": f"Venue {venue_name} not found in database"})
            
        # Check if opponent team exists
        if opponent_team not in ipl_teams:
            return jsonify({"error": f"Team {opponent_team} not found in database"})
        
        # Get batter profile
        batter = batter_profiles[batter_name]
        
        # Get venue difficulty
        venue_difficulty = ipl_venues[venue_name]["venue_difficulty"]
        
        # Get opponent team bowling strength
        opponent_strength = team_bowling_strengths[opponent_team]
        
        # Get batter's average against this team
        average_runs_vs_team = batter["team_averages"].get(opponent_team, 35.0)  # Default if not available
        
        # Get batter's average at this venue
        average_runs_vs_venue = batter["venue_averages"].get(venue_name, 35.0)  # Default if not available
        
        # Calculate interaction terms
        batter_opponent_interaction = batter["average_runs"] * opponent_strength
        batter_venue_interaction = batter["average_runs"] * venue_difficulty
        
        # Create feature vector for prediction
        features = [
            batter["average_runs"],
            batter["average_strike_rate"],
            batter["total_balls_faced"],
            batter["total_dismissals"],
            opponent_strength,
            venue_difficulty,
            batter_opponent_interaction,
            batter_venue_interaction,
            average_runs_vs_team,
            average_runs_vs_venue
        ]
        
        # Convert to numpy array for prediction
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return results
        return jsonify({
            "batter": batter_name,
            "opponent_team": opponent_team,
            "venue": venue_name,
            "predicted_runs": float(prediction[0]),
            "features_used": {
                "average_runs": float(batter["average_runs"]),
                "average_strike_rate": float(batter["average_strike_rate"]),
                "total_balls_faced": int(batter["total_balls_faced"]),
                "total_dismissals": int(batter["total_dismissals"]),
                "opponent_strength": float(opponent_strength),
                "venue_difficulty": float(venue_difficulty),
                "batter_opponent_interaction": float(batter_opponent_interaction),
                "batter_venue_interaction": float(batter_venue_interaction),
                "average_runs_vs_team": float(average_runs_vs_team),
                "average_runs_vs_venue": float(average_runs_vs_venue)
            }
        })

    except Exception as e:
        import traceback
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



#End of Code... Thank U..