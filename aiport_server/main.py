from flask import Blueprint, jsonify, render_template, request

main_router = Blueprint('main', __name__)

@main_router.route('/')
def router_main():
    return render_template(
        'index.html', 
        info={
            'Name': __name__,
            'Methods': ['GET']
        }
    )