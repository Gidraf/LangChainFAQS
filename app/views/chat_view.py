import os
import subprocess
import uuid
from flask import request,jsonify, abort, render_template, url_for, send_file, render_template, redirect
import simplejson as json
import arrow
from ..model import db
from . import chat
from  ..services.chat_services import process_enquiery


@chat.route("/api/v1/inquiry", methods=["POST"])
def chat_inquiries():
    try:
        session_id = request.json.get("session_id")
        inquiry = request.json.get("inquiry")
        print(session_id)
        print(inquiry)
        msg = process_enquiery(inquiry, session_id)
        return jsonify({
            "status":"success",
            "message":str(msg)
        }), 200
    except Exception as e:
        print(str(e))
        return jsonify({
            "status":"failed",
            "message":str(e)
        }), 200