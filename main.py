import pickle
import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)

# Global variables
engine = None
Vectorize = None
Tfvect = None

# Get port from environment variable, default to 8080
PORT = int(os.environ.get("PORT", 8080))

# Check if data files exist, if not create them
def prepare_data_if_needed():
    global engine, Vectorize, Tfvect
    
    try:
        if not os.path.exists('data'):
            os.makedirs('data', exist_ok=True)
            
        # Only prepare data if any of the required files are missing
        if (not os.path.exists('data/vectorizer.pkl') or 
            not os.path.exists('data/tfidf_matrix.pkl') or 
            not os.path.exists('data/engine.pkl')):
            
            print("Data files not found. Preparing data...")
            
            # Load and process book data
            print("Loading book data...")
            data_sources = [
                "./books.csv",
                "/kaggle/input/book-recommendation-good-book-api/books.csv",
                "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
            ]
            
            df = None
            for source in data_sources:
                try:
                    print(f"Trying to load data from: {source}")
                    if source.startswith(('http://', 'https://')):
                        df = pd.read_csv(source)
                    else:
                        if os.path.exists(source):
                            df = pd.read_csv(source, on_bad_lines='skip')
                    if df is not None:
                        print(f"Successfully loaded data from {source}")
                        break
                except Exception as e:
                    print(f"Could not load from {source}: {e}")
            
            if df is None:
                raise Exception("Could not load book data from any source")
            
            # Create a copy of the dataframe to match the notebook
            engine = pd.DataFrame(df.copy())
            
            # Add the rating column (same as average_rating)
            engine["rating"] = engine["average_rating"]
            
            # Clean the titles for better matching - exactly as in notebook
            engine["Re_title"] = engine["title"].str.replace("[^a-zA-Z0-9]", " ", regex=True)
            
            # Create the TF-IDF vectors - exactly as in notebook
            print("Creating TF-IDF vectors...")
            Vectorize = TfidfVectorizer()
            Tfvect = Vectorize.fit_transform(engine["Re_title"])
            
            # Save the vectorizer and matrix
            print("Saving vectorizer and TF-IDF matrix...")
            with open('data/vectorizer.pkl', 'wb') as f:
                pickle.dump(Vectorize, f)
            
            with open('data/tfidf_matrix.pkl', 'wb') as f:
                pickle.dump(Tfvect, f)
            
            # Save the engine dataframe
            print("Saving book engine...")
            engine.to_pickle('data/engine.pkl')
                
            print("Data preparation complete!")
            return True
        else:
            # If files exist, load them
            print("Loading existing data files...")
            with open('data/vectorizer.pkl', 'rb') as f:
                Vectorize = pickle.load(f)
                
            with open('data/tfidf_matrix.pkl', 'rb') as f:
                Tfvect = pickle.load(f)
            
            engine = pd.read_pickle('data/engine.pkl')
            print("Existing data files loaded successfully!")
            return True
    except Exception as e:
        print(f"Error in prepare_data_if_needed: {e}")
        return False

# Call prepare data before starting the app
if not prepare_data_if_needed():
    print("ERROR: Failed to prepare or load data. Exiting.")
    import sys
    sys.exit(1)

# Define a better search function that balances relevance with rating
def search(Query, Vectorize_param=None):
    global engine, Tfvect, Vectorize
    
    # Use provided Vectorize or global one
    if Vectorize_param is not None:
        vec = Vectorize_param
    else:
        vec = Vectorize
    
    # Safety check
    if engine is None or Tfvect is None or vec is None:
        raise ValueError("Search engine components not initialized")
        
    # Process query in the same way as during training
    sub_match = re.sub("[^a-zA-Z0-9]"," ", Query.lower())
    Query_vec = vec.transform([sub_match])
    
    # Calculate similarity scores
    similarity = cosine_similarity(Query_vec, Tfvect).flatten()
    
    # Create a combined score that weighs both similarity and rating
    # This ensures results are relevant to the query, not just high-rated
    engine_copy = engine.copy()
    engine_copy['similarity'] = similarity
    
    # Find books with non-zero similarity (actually match the query)
    relevant_results = engine_copy[engine_copy['similarity'] > 0]
    
    # If no relevant results, fall back to top similarity scores
    if len(relevant_results) < 5:
        # Get top results by similarity only
        top_indices = np.argsort(similarity)[-10:][::-1]
        results = engine.iloc[top_indices]
        return results.head(5)
    
    # Sort by a balanced score (similarity * rating)
    # This ensures books are both relevant AND well-rated
    relevant_results['balanced_score'] = relevant_results['similarity'] * relevant_results['average_rating']
    final_results = relevant_results.sort_values('balanced_score', ascending=False)
    
    return final_results.head(5)

@app.route('/search', methods=['GET'])
def search_api():
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    try:
        # Use the search function from the notebook
        results = search(query)
        
        # Convert DataFrame results to JSON
        json_results = []
        for _, book in results.iterrows():
            json_results.append({
                "title": book.get("title", ""),
                "authors": book.get("authors", ""),
                "average_rating": float(book.get("average_rating", 0)),
                "similarity_score": float(book.get("similarity", 0) if "similarity" in book else 0),
                "publisher": book.get("publisher", ""),
                "language": book.get("language_code", "")
            })
        
        return jsonify({"results": json_results, "query": query})
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    global engine
    if engine is not None:
        return jsonify({"status": "healthy", "book_count": len(engine)})
    else:
        return jsonify({"status": "unhealthy", "reason": "Data not loaded"}), 500

@app.route('/debug', methods=['GET'])
def debug():
    query = request.args.get('q', 'harry potter')
    limit = int(request.args.get('limit', '10'))
    
    try:
        # Process query
        sub_match = re.sub("[^a-zA-Z0-9]"," ", query.lower())
        query_vec = Vectorize.transform([sub_match])
        similarity = cosine_similarity(query_vec, Tfvect).flatten()
        
        # Get indices sorted by similarity
        indices = np.argsort(similarity)[-limit:][::-1]
        
        # Get results
        results = []
        for idx in indices:
            book = engine.iloc[idx]
            results.append({
                "title": book.get("title", ""),
                "authors": book.get("authors", ""),
                "average_rating": float(book.get("average_rating", 0)),
                "similarity": float(similarity[idx]),
                "rating*similarity": float(similarity[idx] * book.get("average_rating", 0)),
            })
        
        return jsonify({
            "query": query,
            "processed_query": sub_match,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": f"Debug failed: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({"status": "Book Recommendation API is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT) 