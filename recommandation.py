from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import hashlib

class ProductRecommender:
    def __init__(self, excel_file_path='Apparels_shared.xlsx', embeddings_dir='embeddings_cache'):
        """Initialize the recommender with product data and embeddings."""
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(exist_ok=True)
        
        self.excel_file_path = excel_file_path
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.df = self._load_catalog(excel_file_path)
        
        # Create a hash of the DataFrame to detect changes
        self.df_hash = self._create_df_hash()
        
        # Check if embeddings exist and are up to date
        if self._embeddings_exist() and self._embeddings_up_to_date():
            print("Loading existing embeddings...")
            self._load_embeddings()
        else:
            print("Building new embeddings (this will take a moment)...")
            self._build_and_save_embeddings()
    
    def _create_df_hash(self):
        """Create a hash of the DataFrame content to detect changes."""
        # Convert DataFrame to string and hash it
        df_string = pd.util.hash_pandas_object(self.df, index=True).values
        return hashlib.md5(str(df_string).encode()).hexdigest()
    
    def _embeddings_exist(self):
        """Check if embedding files exist."""
        required_files = [
            self.embeddings_dir / 'product_embeddings.npy',
            self.embeddings_dir / 'faiss_index.bin',
            self.embeddings_dir / 'descriptions.pkl',
            self.embeddings_dir / 'metadata.pkl'
        ]
        return all(f.exists() for f in required_files)
    
    def _embeddings_up_to_date(self):
        """Check if embeddings are up to date with current data."""
        try:
            metadata_file = self.embeddings_dir / 'metadata.pkl'
            
            # Load metadata to check if DataFrame content matches
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Check if DataFrame shape and content hash match
            return (metadata.get('df_shape') == self.df.shape and 
                    metadata.get('df_hash') == self.df_hash)
        except Exception as e:
            print(f"Error checking embeddings: {e}")
            return False
    
    def _load_catalog(self, file_path):
        """Load product catalog from Excel file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        return pd.read_excel(file_path)
    
    def _describe_product(self, row):
        """Create a clean, natural-language description of a product from its attributes (excluding price)."""
        parts = []
        attr_labels = [
            ('name', 'Name'),
            ('category', 'Category'),
            ('fit', 'Fit'),
            ('fabric', 'Fabric'),
            ('available_sizes', 'Available Sizes'), 
            ('sleeve_length', 'Sleeve Length'),
            ('color_or_print', 'Color/Print'),
            ('occasion', 'Occasion'),
            ('neckline', 'Neckline'),
            ('length', 'Length'),
            ('pant_type', 'Pant Type')
        ]
        for attr, label in attr_labels:
            value = row.get(attr)
            if pd.notna(value) and value is not None and str(value).strip():
                if isinstance(value, list):
                    value_str = ', '.join(str(v) for v in value if v is not None and str(v).strip())
                else:
                    value_str = str(value).strip()
                parts.append(f"{label}: {value_str}")
        return ". ".join(parts)

    def _build_and_save_embeddings(self):
        """Build embeddings and save them to disk."""
        # Generate descriptions for all products
        print("Generating product descriptions...")
        self.df['description'] = self.df.apply(self._describe_product, axis=1)
        descriptions = self.df['description'].tolist()
        
        # Create embeddings
        print("Encoding product descriptions...")
        self.product_embeddings = self.model.encode(
            descriptions, 
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Build FAISS index
        print("Building FAISS index...")
        d = self.product_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.product_embeddings.astype('float32'))
        
        # Save everything
        self._save_embeddings()
        print(f"Embeddings saved to {self.embeddings_dir}")
    
    def _save_embeddings(self):
        """Save embeddings and related data to disk."""
        try:
            # Save embeddings
            np.save(self.embeddings_dir / 'product_embeddings.npy', self.product_embeddings)
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.embeddings_dir / 'faiss_index.bin'))
            
            # Save descriptions
            with open(self.embeddings_dir / 'descriptions.pkl', 'wb') as f:
                pickle.dump(self.df['description'].tolist(), f)
            
            # Save metadata including content hash
            metadata = {
                'df_shape': self.df.shape,
                'df_hash': self.df_hash,
                'embedding_dim': self.product_embeddings.shape[1],
                'model_name': 'all-MiniLM-L6-v2',
                'created_at': pd.Timestamp.now().isoformat()
            }
            with open(self.embeddings_dir / 'metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
                
        except Exception as e:
            print(f"Error saving embeddings: {e}")
            raise
    
    def _load_embeddings(self):
        """Load embeddings and related data from disk."""
        try:
            # Load embeddings
            self.product_embeddings = np.load(self.embeddings_dir / 'product_embeddings.npy')
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.embeddings_dir / 'faiss_index.bin'))
            
            # Load descriptions (and add to DataFrame if not present)
            with open(self.embeddings_dir / 'descriptions.pkl', 'rb') as f:
                descriptions = pickle.load(f)
            
            if 'description' not in self.df.columns:
                self.df['description'] = descriptions
                
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            # If loading fails, rebuild embeddings
            self._build_and_save_embeddings()
    
    def _query_to_description(self, structured_attrs):
        """Convert structured attributes to a text description for embedding (excluding price)."""
        def format_value(value):
            if isinstance(value, list):
                return [str(v).strip() for v in value if v is not None and str(v).strip()]
            if value is not None and str(value).strip():
                return str(value).strip()
            return None
        query_row = {
            'name': None,
            'category': format_value(structured_attrs.get('category')),
            'fit': format_value(structured_attrs.get('fit')),
            'fabric': format_value(structured_attrs.get('fabric')),
            'available_sizes': format_value(structured_attrs.get('size')),
            'sleeve_length': format_value(structured_attrs.get('sleeve_length')),
            'color_or_print': format_value(structured_attrs.get('color_or_print')),
            'occasion': format_value(structured_attrs.get('occasion')),
            'neckline': format_value(structured_attrs.get('neckline')),
            'length': format_value(structured_attrs.get('length')),
            'pant_type': format_value(structured_attrs.get('pant_type'))
            # price intentionally excluded
        }
        return self._describe_product(query_row)
    
    def recommend_products(self, structured_attrs, k=10, final_k=3):
        """
        Hybrid recommendation engine:
        1. Get top k candidates using semantic similarity
        2. Apply improved rules to select final_k products
        """
        try:
            # Convert structured attributes to text description
            query_description = self._query_to_description(structured_attrs)
            if not query_description.strip():
                print("No valid attributes found, returning random sample")
                return self.df.sample(min(final_k, len(self.df)))
            print(f"Query description: {query_description}")
            # Create embedding for the query
            query_embedding = self.model.encode(query_description, normalize_embeddings=True)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            # Search for similar products (get more candidates for filtering)
            distances, indices = self.index.search(query_embedding, min(k, len(self.df)))
            # Get top k recommended products
            candidates = self.df.iloc[indices[0]].copy()
            candidates['similarity_score'] = 1 - distances[0]  # Convert distance to similarity
            # Apply improved hybrid scoring with partial/fuzzy rules
            final_recommendations = self._apply_hybrid_scoring(candidates, structured_attrs, final_k)
            return final_recommendations
        except Exception as e:
            print(f"Error in hybrid recommendation: {e}")
            # Fallback: return random sample
            return self.df.sample(min(final_k, len(self.df)))

    def _apply_hybrid_scoring(self, candidates, structured_attrs, final_k):
        """
        Apply improved rules and scoring to select final recommendations from candidates.
        Includes partial/fuzzy matching for color, fabric, etc.
        """
        import difflib
        scored_candidates = candidates.copy()
        scored_candidates['price_score'] = 0.0
        scored_candidates['size_score'] = 0.0
        scored_candidates['occasion_score'] = 0.0
        scored_candidates['color_score'] = 0.0
        scored_candidates['category_score'] = 0.0
        scored_candidates['fabric_score'] = 0.0
        # Rule 1: Price Matching (25% weight)
        query_price = structured_attrs.get('price_max') or structured_attrs.get('price')
        if query_price is not None and 'price' in scored_candidates.columns:
            try:
                query_price = float(query_price)
                scored_candidates['price'] = pd.to_numeric(scored_candidates['price'], errors='coerce')
                def calculate_price_score(price):
                    if pd.isna(price):
                        return 0.0
                    price_diff = abs(float(price) - query_price)
                    max_diff = query_price * 0.5
                    return max(0, 1 - (price_diff / max_diff)) if max_diff > 0 else 1.0
                scored_candidates['price_score'] = scored_candidates['price'].apply(calculate_price_score)
            except Exception as e:
                print(f"Error in price scoring: {e}")
        # Rule 2: Size Matching (20% weight, partial match allowed)
        query_sizes = structured_attrs.get('size')
        if query_sizes and 'available_sizes' in scored_candidates.columns:
            if isinstance(query_sizes, str):
                query_sizes = [query_sizes]
            def calculate_size_score(available_sizes):
                if pd.isna(available_sizes):
                    return 0.0
                available = str(available_sizes).lower().split(',')
                available = [s.strip() for s in available]
                for query_size in query_sizes:
                    if any(query_size.lower() in avail for avail in available):
                        return 1.0
                return 0.0
            scored_candidates['size_score'] = scored_candidates['available_sizes'].apply(calculate_size_score)
        # Rule 3: Occasion Matching (10% weight, partial match)
        query_occasion = structured_attrs.get('occasion')
        if query_occasion and 'occasion' in scored_candidates.columns:
            def calculate_occasion_score(occasion):
                if pd.isna(occasion):
                    return 0.0
                return 1.0 if query_occasion.lower() in str(occasion).lower() else 0.0
            scored_candidates['occasion_score'] = scored_candidates['occasion'].apply(calculate_occasion_score)
        # Rule 4: Color/Print Matching (10% weight, fuzzy match)
        query_color = structured_attrs.get('color_or_print')
        if query_color and 'color_or_print' in scored_candidates.columns:
            def calculate_color_score(color):
                if pd.isna(color):
                    return 0.0
                # Fuzzy match using difflib
                color_str = str(color).lower()
                query_str = str(query_color).lower()
                if query_str in color_str:
                    return 1.0
                # Fuzzy ratio
                ratio = difflib.SequenceMatcher(None, query_str, color_str).ratio()
                return ratio if ratio > 0.5 else 0.0
            scored_candidates['color_score'] = scored_candidates['color_or_print'].apply(calculate_color_score)
        # Rule 5: Category Exact/Partial Match (10% weight)
        query_category = structured_attrs.get('category')
        if query_category and 'category' in scored_candidates.columns:
            if isinstance(query_category, list):
                query_category = query_category[0] if query_category else None
            if query_category:
                def calculate_category_score(category):
                    if pd.isna(category):
                        return 0.0
                    cat_str = str(category).lower()
                    query_str = str(query_category).lower()
                    if query_str == cat_str:
                        return 1.0
                    if query_str in cat_str or cat_str in query_str:
                        return 0.7
                    return 0.0
                scored_candidates['category_score'] = scored_candidates['category'].apply(calculate_category_score)
        # Rule 6: Fabric Preference (10% weight, fuzzy/partial)
        query_fabric = structured_attrs.get('fabric')
        if query_fabric and 'fabric' in scored_candidates.columns:
            if isinstance(query_fabric, list):
                query_fabric = query_fabric[0] if query_fabric else None
            if query_fabric:
                def calculate_fabric_score(fabric):
                    if pd.isna(fabric):
                        return 0.0
                    fabric_str = str(fabric).lower()
                    query_str = str(query_fabric).lower()
                    if query_str in fabric_str:
                        return 1.0
                    ratio = difflib.SequenceMatcher(None, query_str, fabric_str).ratio()
                    return ratio if ratio > 0.5 else 0.0
                scored_candidates['fabric_score'] = scored_candidates['fabric'].apply(calculate_fabric_score)
        # Calculate weighted final score
        weights = {
            'similarity_score': 0.25,  # 25% semantic similarity
            'price_score': 0.25,      # 25% price matching
            'size_score': 0.2,        # 20% size availability
            'occasion_score': 0.1,    # 10% occasion matching
            'color_score': 0.1,       # 10% color matching
            'category_score': 0.05,   # 5% category match
            'fabric_score': 0.05      # 5% fabric preference
        }
        scored_candidates['hybrid_score'] = 0.0
        for score_type, weight in weights.items():
            if score_type in scored_candidates.columns:
                scored_candidates['hybrid_score'] += scored_candidates[score_type] * weight
        # Apply strict filters (must-have requirements)
        filtered_candidates = scored_candidates.copy()
        # Strict Filter 1: Size availability (if specified)
        query_sizes = structured_attrs.get('size')
        if query_sizes:
            filtered_candidates = filtered_candidates[filtered_candidates['size_score'] > 0]
            if len(filtered_candidates) == 0:
                filtered_candidates = scored_candidates.copy()
        # Strict Filter 2: Price range (if specified with max budget)
        if query_price and 'price' in filtered_candidates.columns:
            max_budget = query_price * 1.2
            filtered_candidates = filtered_candidates[
                (pd.isna(filtered_candidates['price'])) |
                (filtered_candidates['price'] <= max_budget)
            ]
            if len(filtered_candidates) == 0:
                filtered_candidates = scored_candidates.copy()
        # Sort by hybrid score and return top final_k
        final_recommendations = filtered_candidates.sort_values(
            ['hybrid_score', 'similarity_score'],
            ascending=[False, False]
        ).head(final_k)
        # Create clean display dataframe with only essential information
        display_columns = {
            'name': 'Product Name',
            'category': 'Category',
            'price': 'Price ($)',
            'color_or_print': 'Color/Print',
            'fabric': 'Fabric',
            'available_sizes': 'Available Sizes',
            'occasion': 'Occasion',
            'hybrid_score': 'Match Score'
        }
        clean_recommendations = final_recommendations.copy()
        available_columns = {k: v for k, v in display_columns.items()
                            if k in clean_recommendations.columns}
        display_df = clean_recommendations[list(available_columns.keys())].copy()
        display_df = display_df.rename(columns=available_columns)
        if 'Price ($)' in display_df.columns:
            display_df['Price ($)'] = display_df['Price ($)'].apply(
                lambda x: f"${x:.0f}" if pd.notna(x) else "Price not available"
            )
        if 'Match Score' in display_df.columns:
            display_df['Match Score'] = display_df['Match Score'].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "0%"
            )
        # Add ranking
        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        # Add Final Score as a percentage (hybrid_score)
        if 'Match Score' in display_df.columns:
            display_df['Final Score'] = display_df['Match Score']
        # Clean up any NaN values for display
        display_df = display_df.fillna("Not specified")
        print(f"Selected {len(display_df)} products for display")
        return display_df
