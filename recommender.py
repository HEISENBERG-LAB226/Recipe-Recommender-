import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RecipeRecommender:
    def __init__(self, json_path):
        """
        Initialize the Recipe Recommender System
        
        Args:
            json_path: Path to the train.json file
        """
        # Load JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        recipes = []
        for item in data:
            recipe_id = item.get('id', 'Unknown')
            cuisine = item.get('cuisine', 'Unknown')
            ingredients_list = item.get('ingredients', [])
            
            # Join ingredients into a single string
            ingredients_str = ' '.join(ingredients_list)
            
            # Detect allergens automatically
            allergens = self._detect_allergens(ingredients_list)
            
            recipes.append({
                'id': recipe_id,
                'recipe': f"{cuisine.replace('_', ' ').title()} Recipe #{recipe_id}",
                'cuisine': cuisine,
                'ingredients': ingredients_str,
                'ingredients_list': ingredients_list,
                'allergens': allergens
            })
        
        self.df = pd.DataFrame(recipes)
        
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=500
        )
        
        # Fit the vectorizer on all recipe ingredients
        self.recipe_vectors = self.vectorizer.fit_transform(self.df['ingredients'])
    
    def _detect_allergens(self, ingredients_list):
        """
        Automatically detect allergens from ingredients list
        
        Args:
            ingredients_list: List of ingredient strings
            
        Returns:
            String of detected allergens or 'none'
        """
        allergens = set()
        
        # Convert all ingredients to lowercase for matching
        ingredients_lower = [ing.lower() for ing in ingredients_list]
        ingredients_str = ' '.join(ingredients_lower)
        
        # Egg allergens
        egg_keywords = ['egg', 'eggs', 'mayo', 'mayonnaise']
        if any(keyword in ingredients_str for keyword in egg_keywords):
            allergens.add('eggs')
        
        # Dairy allergens
        dairy_keywords = ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'yoghurt', 
                         'dairy', 'parmesan', 'mozzarella', 'cheddar', 'feta']
        if any(keyword in ingredients_str for keyword in dairy_keywords):
            allergens.add('dairy')
        
        # Nut allergens
        nut_keywords = ['nut', 'nuts', 'peanut', 'almond', 'cashew', 'walnut', 
                       'pecan', 'pistachio', 'hazelnut']
        if any(keyword in ingredients_str for keyword in nut_keywords):
            allergens.add('nuts')
        
        # Soy allergens
        soy_keywords = ['soy', 'tofu', 'edamame', 'miso']
        if any(keyword in ingredients_str for keyword in soy_keywords):
            allergens.add('soy')
        
        return ', '.join(sorted(allergens)) if allergens else 'none'
    
    def recommend(self, user_ingredients, top_n=5, exclude_allergens=None):
        """
        Recommend recipes based on user ingredients
        
        Args:
            user_ingredients: String of comma-separated ingredients
            top_n: Number of recommendations to return
            exclude_allergens: List of allergens to exclude
            
        Returns:
            List of dictionaries containing recipe information
        """
        if exclude_allergens is None:
            exclude_allergens = []
        
        # Clean and prepare user input
        user_ingredients = user_ingredients.lower().strip()
        
        # Transform user ingredients to vector
        user_vector = self.vectorizer.transform([user_ingredients])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, self.recipe_vectors).flatten()
        
        # Add similarity scores to dataframe
        df_with_scores = self.df.copy()
        df_with_scores['similarity'] = similarities
        
        # Filter out allergens if specified
        if exclude_allergens:
            for allergen in exclude_allergens:
                df_with_scores = df_with_scores[
                    ~df_with_scores['allergens'].str.contains(allergen, case=False, na=False)
                ]
        
        # Sort by similarity and get top N
        top_recipes = df_with_scores.nlargest(top_n, 'similarity')
        
        # Format results
        results = []
        for _, row in top_recipes.iterrows():
            results.append({
                'id': row['id'],
                'recipe': row['recipe'],
                'cuisine': row['cuisine'],
                'ingredients': row['ingredients'],
                'ingredients_list': row['ingredients_list'],
                'allergens': row['allergens'],
                'similarity': row['similarity']
            })
        
        return results
    
    def get_all_recipes(self):
        """Return all recipes in the dataset"""
        return self.df.to_dict('records')
    
    def search_by_cuisine(self, cuisine_name):
        """Search for recipes by cuisine type"""
        matches = self.df[self.df['cuisine'].str.contains(cuisine_name, case=False, na=False)]
        return matches.to_dict('records')
