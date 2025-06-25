import re
from typing import List, Dict, Pattern

# 1) Sample text for building vocabulary and testing
text = "IdeaWeaver-- a comprehensive CLI tool for AI model training and evaluation?"

# 2) Initial tokenization to extract all tokens
raw_tokens = re.split(r'([,.:;?_!"\'()]|--|\s)', text)
tokens     = [tok.strip() for tok in raw_tokens if tok.strip()]

# 3) Build a sorted, unique vocabulary mapping token → ID
all_tokens = sorted(list(set(tokens)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# 4) Define the BasicTokenizer class
class BasicTokenizer:
    def __init__(
        self,
        token_index: Dict[str, int],
        split_pattern: Pattern = re.compile(r'([,.:;?_!"\'()]|--|\s)'),
        rejoin_pattern: Pattern = re.compile(r'\s+([,.:;?_!"\'()])')
    ):
        # Step 1: Store the vocabulary
        self.token_index     = token_index
        # Step 2: Create inverse mapping for decoding
        self.index_token     = {idx: tok for tok, idx in token_index.items()}
        # Regex patterns for tokenize & re-join
        self._split_pattern  = split_pattern
        self._rejoin_pattern = rejoin_pattern
        # Optional ID for unknown tokens
        self.unknown_id      = token_index.get("<|unk|>", None)

    def _tokenize(self, text: str) -> List[str]:
        # Split on punctuation, double-dash, or whitespace, then clean up
        raw = self._split_pattern.split(text)
        return [piece.strip() for piece in raw if piece.strip()]

    def encode(self, text: str) -> List[int]:
        # Step 3: Turn text into a list of token IDs
        tokens = self._tokenize(text)
        ids = []
        for tok in tokens:
            if tok in self.token_index:
                ids.append(self.token_index[tok])
            elif self.unknown_id is not None:
                ids.append(self.unknown_id)
            # otherwise skip
        return ids

    def decode(self, ids: List[int]) -> str:
        # Step 4: Convert IDs back into tokens and join
        tokens = [ self.index_token[i] for i in ids if i in self.index_token ]
        output = ""
        for tok in tokens:
            # if it's pure punctuation, append without space
            if re.fullmatch(r'[,.:;?_!"\'()\-\–]+', tok):
                output += tok
            else:
                # otherwise prepend a space (unless first token)
                if output:
                    output += " "
                output += tok
        # Step 5 is implicit here—spaces before punctuation never get added
        return output

# 5) Instantiate and test
tokenizer = BasicTokenizer(vocab)
print(f"Unknown token ID: {tokenizer.unknown_id}")

# Encode the original text
ids = tokenizer.encode(text)
print("Token IDs:", ids)
# ➞ e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Decode back to string
decoded_text = tokenizer.decode(ids)
print("Decoded text:", decoded_text)
# ➞ "IdeaWeaver-- a comprehensive CLI tool for AI model training and evaluation?"

# Test with new text
text = "Hello, how are you?"
print(f"\nTesting with new text: '{text}'")

# Debug: Show tokenization process
debug_tokens = tokenizer._tokenize(text)
print(f"Debug - Tokens found: {debug_tokens}")
print(f"Debug - Number of tokens: {len(debug_tokens)}")

encoded_ids = tokenizer.encode(text)
print("Encoded IDs:", encoded_ids)
decoded_new = tokenizer.decode(encoded_ids)
print("Decoded text:", decoded_new) 
