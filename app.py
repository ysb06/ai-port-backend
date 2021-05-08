from flask import Flask, render_template, request

from bert_chat.router import chat_router

app = Flask(__name__)
app.register_blueprint(chat_router, url_prefix="/chat")


@app.route("/")
def render_info_page():
    return render_template("index.html", name=__name__)


@app.route("/translation/kren", methods=("GET", "POST", "PUT", "DELETE"))
def route_ke_translation():
    if request.method == "GET":
        return str(request.method)
    else:
        return str(request.method)


if __name__ == "__main__":
    app.run(debug=True)
