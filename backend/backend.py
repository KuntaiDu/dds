import os
import logging
from flask import Flask, request, jsonify
from dds_utils import ServerConfig
import json
from .server import Server

app = Flask(__name__)
server = None


@app.route("/")
@app.route("/index")
def index():
    # TODO: Add debugging information to the page if needed
    return "Much to do!"


@app.route("/init", methods=["POST"])
def initialize_server():
    d = request.args
    new_config = ServerConfig(
        float(d.get("low_resolution")), float(d.get("high_threshold")),
        int(d.get("low_qp")), int(d.get("high_qp")),
        int(d.get("batch_size")), float(d.get("high_threshold")),
        float(d.get("low_threshold")), float(d.get("max_object_size")),
        None, float(d.get("tracker_length")),  # Minimum object size is none temporarily
        float(d.get("boundary")), float(d.get("intersection_threshold")),
        float(d.get("tracking_threshold")),
        float(d.get("suppression_threshold")), bool(d.get("simulation")),
        float(d.get("rpn_enlarge_ratio")), float(d.get("prune_score")),
        float(d.get("objfilter_iou")), float(d.get("size_obj")))
    global server
    if not server:
        logging.basicConfig(
            format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
            level="INFO")
        server = Server(new_config, int(d.get("nframes")))
        os.makedirs("server_temp", exist_ok=True)
        os.makedirs("server_temp-cropped", exist_ok=True)
        return "New Init"
    else:
        server.reset_state(int(d.get("nframes")))
        return "Reset"


@app.route("/low", methods=["POST"])
def low_query():
    file_data = request.files["media"]
    results = server.perform_low_query(file_data)

    return jsonify(results)


@app.route("/high", methods=["POST"])
def high_query():
    file_data = request.files["media"]
    results = server.perform_high_query(file_data)

    return jsonify(results)
