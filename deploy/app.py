import json
import os
from flask import Flask, jsonify,request,send_from_directory
from flask_cors import CORS
import urllib.parse
import uuid
import requests
from werkzeug.utils import secure_filename
from utils.det import det
from utils.warp import get_warp,get_single
from utils.rec import rec_bp
app = Flask(__name__,static_folder='dist')
CORS(app)
app.register_blueprint(rec_bp)
UPLOADS_FOLDER = './dist/uploads/'

@app.route('/',methods=['GET'])
def index():
    return send_from_directory(app.static_folder,'index.html')

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/upload', methods=['POST'])
def upload():
    os.chdir("E:\\vv\\学校\\大三下\\深度学习\\课程设计\\Handwritten_Chinese_recognition\\deploy")
    file = request.files['image']
    file_extension = os.path.splitext(file.filename)[1]
    unique_id = str(uuid.uuid4())
    save_name = f'{unique_id}{file_extension}'
    safename = secure_filename(save_name[-8:])
    file.save(os.path.join(UPLOADS_FOLDER, safename))
    file.seek(0)

    return f'http://127.0.0.1:5000/uploads/{safename}'

@app.route('/api/rec',methods=['POST'])
def rec():
    os.chdir('./utils')
    url = request.json['picsrc']
    def get_url_basename(url):
        parsed_url = urllib.parse.urlparse(url)
        basename = os.path.basename(parsed_url.path)
        return basename
    picname = get_url_basename(url)
    print(picname)
    respic,boxs = det(picname)
    warps = get_warp(picname,boxs)
    lines = get_single(warps=warps)
    # 将每一行转成文字：
    sentences = []
    headers = {
    'Content-Type': 'application/json',
    }
    for line in lines:
        data = {"picnames":line} 
        res = requests.post('http://127.0.0.1:5000/api/rec_single',json=data,headers=headers)
        res_dict = json.loads(res.text)
        words = ''.join(res_dict["words"])
        sentences.append(words)
    return jsonify({
        "respic":f'http://127.0.0.1:5000/uploads/{respic}',
        "warps":sentences
    })
 

if __name__ == '__main__':
    app.run(port=5000,debug=True)