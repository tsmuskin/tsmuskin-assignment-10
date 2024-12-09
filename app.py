from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
from image_search import search_with_image, search_with_text, search_combined, initialize_pca

app = Flask(__name__)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the search query
@app.route('/search', methods=['POST'])
def search():
    print(request.form)
    query_type = request.form.get('query_type')
    text_query = request.form.get('text_query', None)
    
    image_file = request.files.get('image_query', None)
    weight = float(request.form.get('weight', 0.5))  # Default weight for combined queries
    use_pca = request.form.get('use_pca', 'false') == 'true'
    k_components = int(request.form.get('k_components', 5))  # Default to 5 PCA components

    print("use_pca: ", use_pca)
    print(k_components)

    # Handle text-only, image-only, and combined queries
    results = []
    if query_type == 'text':
        results = search_with_text(text_query)
        print(results)
    elif query_type == 'image':
        if image_file:
            image_path = os.path.join('uploads', image_file.filename)
            image_file.save(image_path)
            results = search_with_image(image_path, use_pca=use_pca, k_components=k_components)
            print(results)

    elif query_type == 'combined':
        if image_file and text_query:
            image_path = os.path.join('uploads', image_file.filename)
            image_file.save(image_path)
            results = search_combined(text_query, image_path, weight, use_pca, k_components)
            print(results)
    
    response = [
        {"image": url_for('static', filename=f'coco_images_resized/{result[0]}'), "score": result[1]}
        for result in results
    ]
    print(response)
    return jsonify(response)

# Route to serve uploaded and result images
@app.route('/uploads/<filename>')
def uploads(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
