from flask import Flask, request, render_template
from utils import get_most_similar_dog
import os
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    user_image = None
    if request.method == 'POST':
        img = request.files['image']
        # 이미지를 읽고 Base64로 인코딩
        img_bytes = img.read()
        user_image = base64.b64encode(img_bytes).decode('utf-8')
        # 파일 포인터를 다시 처음으로 이동
        img.seek(0)
        result = get_most_similar_dog(img)
    return render_template('index.html', result=result, user_image=user_image)

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)