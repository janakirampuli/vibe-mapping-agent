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
    
    def _describe_product(self, attrs):
        """Create a value-only description of a product/query from its attributes (excluding price and labels), matching the notebook logic."""
        attr_order = [
            'category',
            'size',
            'fit',
            'fabric',
            'sleeve_length',
            'color_or_print',
            'occasion',
            'neckline',
            'length',
            'pant_type'
        ]
        parts = []
        for attr in attr_order:
            value = attrs.get(attr)
            if pd.notna(value) and value is not None and str(value).strip():
                if isinstance(value, list):
                    value_str = ', '.join(str(v) for v in value if v is not None and str(v).strip())
                else:
                    value_str = str(value).strip()
                parts.append(value_str)
        return ". ".join(parts)

    def _row_to_attr_dict(self, row):
        """Convert a product row (Series) to a dict with the same keys/order as used in the notebook embedding."""
        return {
            'category': row.get('category'),
            'size': row.get('available_sizes'),
            'fit': row.get('fit'),
            'fabric': row.get('fabric'),
            'sleeve_length': row.get('sleeve_length'),
            'color_or_print': row.get('color_or_print'),
            'occasion': row.get('occasion'),
            'neckline': row.get('neckline'),
            'length': row.get('length'),
            'pant_type': row.get('pant_type')
        }

    def _build_and_save_embeddings(self):
        """Build embeddings and save them to disk, using consistent description logic for all products."""
        print("Generating product descriptions...")
        # Use consistent dict conversion for each row
        self.df['description'] = self.df.apply(lambda row: self._describe_product(self._row_to_attr_dict(row)), axis=1)
        descriptions = self.df['description'].tolist()
        print("Encoding product descriptions...")
        self.product_embeddings = self.model.encode(
            descriptions, 
            normalize_embeddings=True,
            show_progress_bar=True
        )
        print("Building FAISS index...")
        d = self.product_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.product_embeddings.astype('float32'))
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
    

    def _query_to_description(self, structured_attrs):
        """Convert structured attributes to a text description for embedding (excluding price), using the same logic/order as products and the notebook."""
        attrs = {
            'category': structured_attrs.get('category'),
            'size': structured_attrs.get('size'),
            'fit': structured_attrs.get('fit'),
            'fabric': structured_attrs.get('fabric'),
            'sleeve_length': structured_attrs.get('sleeve_length'),
            'color_or_print': structured_attrs.get('color_or_print'),
            'occasion': structured_attrs.get('occasion'),
            'neckline': structured_attrs.get('neckline'),
            'length': structured_attrs.get('length'),
            'pant_type': structured_attrs.get('pant_type')
        }
        return self._describe_product(attrs)
    
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

        print("Candidates:",  candidates)
        print("structured: ", structured_attrs)
        print("====================== APPLYING HYBRID SCORING ============")
        """
        Apply strict filters (category, size, price, pant_type) and soft scoring for other fields.
        """
        import difflib
        scored_candidates = candidates.copy()
        # --- STRICT FILTERS ---
        # 1. Category (must match if specified)
        query_category = structured_attrs.get('category')
        if query_category and 'category' in scored_candidates.columns:
            if isinstance(query_category, list):
                query_category = query_category[0] if query_category else None
            if query_category:
                scored_candidates = scored_candidates[
                    scored_candidates['category'].str.lower() == str(query_category).lower()
                ]
                if len(scored_candidates) == 0:
                    print("No products match category, relaxing filter")
                    scored_candidates = candidates.copy()
        # 2. Size (must match if specified)
        query_sizes = structured_attrs.get('size')
        if query_sizes and 'available_sizes' in scored_candidates.columns:
            if isinstance(query_sizes, str):
                query_sizes = [query_sizes]
            def size_match(available_sizes):
                if pd.isna(available_sizes):
                    return False
                available = str(available_sizes).lower().split(',')
                available = [s.strip() for s in available]
                for query_size in query_sizes:
                    if any(query_size.lower() in avail for avail in available):
                        return True
                return False
            scored_candidates = scored_candidates[scored_candidates['available_sizes'].apply(size_match)]
            if len(scored_candidates) == 0:
                print("No products match size requirement, relaxing filter")
                scored_candidates = candidates.copy()
        # 3. Price (must be within budget if specified)
        query_price = structured_attrs.get('price_max') or structured_attrs.get('price')
        if query_price and 'price' in scored_candidates.columns:
            try:
                query_price = float(query_price)
                scored_candidates['price'] = pd.to_numeric(scored_candidates['price'], errors='coerce')
                max_budget = query_price * 1.2
                scored_candidates = scored_candidates[(pd.isna(scored_candidates['price'])) | (scored_candidates['price'] <= max_budget)]
                if len(scored_candidates) == 0:
                    print("No products within budget, relaxing price filter")
                    scored_candidates = candidates.copy()
            except Exception as e:
                print(f"Error in price filtering: {e}")
        # 4. Pant Type (if category is pants and pant_type specified)
        query_pant_type = structured_attrs.get('pant_type')
        if query_pant_type and 'pant_type' in scored_candidates.columns and query_category and 'pant' in str(query_category).lower():
            if isinstance(query_pant_type, list):
                query_pant_type = query_pant_type[0] if query_pant_type else None
            if query_pant_type:
                scored_candidates = scored_candidates[
                    scored_candidates['pant_type'].str.lower() == str(query_pant_type).lower()
                ]
                if len(scored_candidates) == 0:
                    print("No products match pant type, relaxing filter")
                    scored_candidates = candidates.copy()
        # --- SOFT SCORING ---
        # Initialize all scores to 0
        for col in ['fit', 'fabric', 'color_or_print', 'occasion', 'sleeve_length', 'neckline', 'length']:
            scored_candidates[f'{col}_score'] = 0.0
        # Fit
        query_fit = structured_attrs.get('fit')
        if query_fit and 'fit' in scored_candidates.columns:
            def fit_score(fit):
                if pd.isna(fit): return 0.0
                return 1.0 if str(query_fit).lower() in str(fit).lower() else 0.0
            scored_candidates['fit_score'] = scored_candidates['fit'].apply(fit_score)
        # Fabric (fuzzy)
        query_fabric = structured_attrs.get('fabric')
        if query_fabric and 'fabric' in scored_candidates.columns:
            if isinstance(query_fabric, list):
                query_fabric = query_fabric[0] if query_fabric else None
            if query_fabric:
                def fabric_score(fabric):
                    if pd.isna(fabric): return 0.0
                    fabric_str = str(fabric).lower()
                    query_str = str(query_fabric).lower()
                    if query_str in fabric_str:
                        return 1.0
                    ratio = difflib.SequenceMatcher(None, query_str, fabric_str).ratio()
                    return ratio if ratio > 0.5 else 0.0
                scored_candidates['fabric_score'] = scored_candidates['fabric'].apply(fabric_score)
        # Color/Print (fuzzy)
        query_color = structured_attrs.get('color_or_print')
        if query_color and 'color_or_print' in scored_candidates.columns:
            def color_score(color):
                if pd.isna(color): return 0.0
                color_str = str(color).lower()
                query_str = str(query_color).lower()
                if query_str in color_str:
                    return 1.0
                ratio = difflib.SequenceMatcher(None, query_str, color_str).ratio()
                return ratio if ratio > 0.5 else 0.0
            scored_candidates['color_or_print_score'] = scored_candidates['color_or_print'].apply(color_score)
        # Occasion
        query_occasion = structured_attrs.get('occasion')
        if query_occasion and 'occasion' in scored_candidates.columns:
            def occasion_score(occasion):
                if pd.isna(occasion): return 0.0
                return 1.0 if str(query_occasion).lower() in str(occasion).lower() else 0.0
            scored_candidates['occasion_score'] = scored_candidates['occasion'].apply(occasion_score)
        # Sleeve Length
        query_sleeve = structured_attrs.get('sleeve_length')
        if query_sleeve and 'sleeve_length' in scored_candidates.columns:
            def sleeve_score(sleeve):
                if pd.isna(sleeve): return 0.0
                return 1.0 if str(query_sleeve).lower() in str(sleeve).lower() else 0.0
            scored_candidates['sleeve_length_score'] = scored_candidates['sleeve_length'].apply(sleeve_score)
        # Neckline
        query_neckline = structured_attrs.get('neckline')
        if query_neckline and 'neckline' in scored_candidates.columns:
            def neckline_score(neck):
                if pd.isna(neck): return 0.0
                return 1.0 if str(query_neckline).lower() in str(neck).lower() else 0.0
            scored_candidates['neckline_score'] = scored_candidates['neckline'].apply(neckline_score)
        # Length
        query_length = structured_attrs.get('length')
        if query_length and 'length' in scored_candidates.columns:
            def length_score(length):
                if pd.isna(length): return 0.0
                return 1.0 if str(query_length).lower() in str(length).lower() else 0.0
            scored_candidates['length_score'] = scored_candidates['length'].apply(length_score)
        # --- Final Score ---
        weights = {
            'similarity_score': 0.25,  # 25% semantic similarity
            'fit_score': 0.15,
            'fabric_score': 0.15,
            'color_or_print_score': 0.1,
            'occasion_score': 0.1,
            'sleeve_length_score': 0.08,
            'neckline_score': 0.07,
            'length_score': 0.05
        }
        scored_candidates['hybrid_score'] = 0.0
        for score_type, weight in weights.items():
            if score_type in scored_candidates.columns:
                scored_candidates['hybrid_score'] += scored_candidates[score_type] * weight
        # Sort by hybrid score and return top final_k
        final_recommendations = scored_candidates.sort_values(
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
            'fit': 'Fit',
            'sleeve_length': 'Sleeve Length',
            'neckline': 'Neckline',
            'length': 'Length',
            'pant_type': 'Pant Type',
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
        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        display_df = display_df.fillna("Not specified")
        print(f"Selected {len(display_df)} products for display")
        return display_df