from flask import request, Blueprint
from flask.helpers import make_response

from bert_chat.chatter import listenChat, initialize

chat_router = Blueprint("chat", __name__)
initialize()


class IncompleteRouteException(Exception):
    def __init__(self) -> None:
        super().__init__("Incomplete route")


@chat_router.route("/", methods=("GET",))
def route_test():
    return "Error"


@chat_router.route("/bert", methods=("GET",))
def route_chat_listen():
    parameter_dict = request.args.to_dict()

    if request.method == "GET":
        if "queryText" in parameter_dict:
            result = listenChat(parameter_dict["queryText"])
            return make_response(result)
        else:
            return make_response("Empty query", 400)
    else:
        raise IncompleteRouteException


@chat_router.route("/bert/train", methods=("POST",))
def route_chat_train():
    parameter_dict = request.args.to_dict()

    if request.method == "POST":
        if "queryText" in parameter_dict:
            result = listenChat(parameter_dict["queryText"])
            return make_response(result)
        else:
            return make_response("Empty query", 400)
    else:
        raise IncompleteRouteException
