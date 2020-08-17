import os
import logging
from flask import Flask, request, jsonify
from dds_utils import ServerConfig
import json
import yaml
from .server import Server

app = Flask(__name__)
server = None

from munch import *


@app.route("/")
@app.route("/index")
def index():
    # TODO: Add debugging information to the page if needed
    return "Much to do!"


@app.route("/initialize_server", methods=["POST"])
def initialize_server():
    args = yaml.load(request.data, Loader=yaml.SafeLoader)
    global server
    if not server:
        logging.basicConfig(
            format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
            level="INFO")
        server = Server(args, args["nframes"])
        return "New Init"
    else:
        server.reset_state(int(args["nframes"]))
        return "Reset"


@app.route("/perform_low_query", methods=["POST"])
def low_query():
    file_data = request.files["media"]
    start_fid = int(request.args['start_fid'])
    end_fid = int(request.args['end_fid'])
    results = server.perform_low_query(start_fid, end_fid, file_data)

    return jsonify(results)

@app.route("/run_inference", methods=["POST"])
def run_inference():
    file_data = request.files["media"]
    results = server.run_inference(file_data)

    return jsonify(results)

@app.route("/perform_high_query", methods=["POST"])
def high_query():
    file_data = request.files["media"]
    json_data = request.files["json"]
    results = server.perform_high_query(file_data, json_data)

    return jsonify(results)
