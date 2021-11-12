#
# Bike sentry back-end service.
# Handles communication with Twilio and front-end where users can add phone numbers.
#
from flask import Flask
from flask import request
from flask_cors import CORS
from flask import jsonify

app = Flask(__name__)
CORS(app)


@app.route("/")
def welcome():
    return "Welcome to bike sentry API!"


# phone numbers must be sanitized and contain no special characters.
@app.route("/addnumber/<tower_id>")
def add_number(tower_id):
    number = request.args.get("number")
    return {"status": "success", "message": f"{number} is now listening to the tower."}


# This will be called by Twilio somehow.
@app.route("/removenumber/<tower_id>")
def remove_number(tower_id):
    number = request.args.get("number")
    return {"status": "success", "message": f"{number} is no longer listening to the tower."}


# Sent from tower, to API to trigger text alerts for a given tower.
# Updates the status of given tower to "Theft Detected".
@app.route("/theft_alert/<tower_id>")
def theft_alert(tower_id):
    return {"status": "success"}


@app.route("/resolve_theft/<tower_id>")
def resolve_theft(tower_id):
    return {"status": "success", "message": f"Theft resolved, Tower {tower_id} is back on sentry mode"}


@app.route("/getsentrytowers")
def get_sentry_towers():
    # convert tree structure of towers to a list
    tower_tree = {"000000": {
        "tower_location": "Testing Room",  # for front end display
        "status": "Offline",  # For front end display
        "tower_id": "000000",
        "tower_name": "Test Sentry",
        "phone_numbers": ["1234567890", "9876543210"]
    }}
    tower_list = []
    for towers_id, tower_dict in tower_tree.items():
        # Don't share phone numbers with front-end
        del tower_dict["phone_numbers"]
        tower_list.append(tower_dict)

    return jsonify(tower_list)
