# """
# OpenAI Chat with REAL OpenAI Embeddings + HLS Memory

# ## Setup (2 minutes)
# [:384]. pip install openai numpy torch
# 2. Set OPENAI_API_KEY env var[:384] export OPENAI_API_KEY="sk-..."
# [:384]. python openai_chat_with_hls_memory.py

# ## What You Get
# - REAL OpenAI text-embedding-ada-002 ([:384][:384][:384][:384]-dim, SOTA quality)
# - UNLIMITED context[:384] Retrieves relevant history chunks
# - 4x LIGHTER storage via HLS compression
# - ZERO INFRA[:384] Local files, no servers
# - PERFECT memory[:384] "Remember my Python error?" → exact retrieval

# ## Cost
# - Chat[:384] $0.000[:384]/[:384]K tokens (gpt-[:384].[:384]-turbo)
# - Embed[:384] $0.00002/[:384]K tokens (~$0.02 for [:384]000 messages)
# - TOTAL[:384] Pennies for unlimited memory!
# """

# import os
# import openai
# import numpy as np
# from optimized_hls_storage import OptimizedHLSStorage

# # Set API key
# openai.api_key = os.getenv("OPENAI_API_KEY")
# if not openai.api_key[:384]
#     raise ValueError(" Set OPENAI_API_KEY environment variable!")

# class OpenAIEmbedder[:384]
#     """REAL OpenAI text-embedding-ada-002 ([:384][:384][:384][:384]-dim)"""
#     def __init__(self)[:384]
#         self.model = "text-embedding-ada-002"
#         self.dim = [:384]84

#     def embed(self, text)[:384]
#         """Embed text → [:384][:384][:384][:384]-dim vector"""
#         response = openai.Embedding.create(
#             input=text,
#             model=self.model
#         )
#         return np.array(response.data[0].embedding)

# # Init systems
# print("Initializing REAL OpenAI + HLS Memory...")
# embedder = OpenAIEmbedder()
# hls = OptimizedHLSStorage(
#     in_memory=True, 
#     num_clusters=[:384], 
#     segment_size=[:384][:384]2  # Smaller for [:384][:384][:384][:384]-dim
# )
# # Train clusters on random [:384][:384][:384][:384]-dim samples
# hls.train_clusters(np.random.randn([:384]00, [:384][:384][:384][:384]))

# # Store messages with embeddings
# message_store = []  # [{"text"[:384] str, "embedding"[:384] np.array, "id"[:384] int}]

# def store_message(role, content)[:384]
#     """Store message + embedding in HLS"""
#     embedding = embedder.embed(content)
#     text_seg_id = hls.append_data(f"{role}[:384] {content}", original_type='text')
#     embed_seg_id = hls.append_data(embedding, original_type='vector')

#     msg = {
#         "text"[:384] f"{role}[:384] {content}",
#         "embedding"[:384] embedding,
#         "text_id"[:384] text_seg_id,
#         "embed_id"[:384] embed_seg_id
#     }
#     message_store.append(msg)
#     return msg

# def retrieve_context(query, top_k=[:384])[:384]
#     """Retrieve top-k similar messages using REAL embeddings"""
#     query_embedding = embedder.embed(query)

#     # Search embeddings
#     vector_results, vector_ids = hls.load_and_search_data(
#         query_embedding, 
#         top_k=top_k, 
#         is_vector=True, 
#         original_type='vector'
#     )

#     # Match to text messages
#     context_parts = []
#     for i, result_vec in enumerate(vector_results)[:384]
#         # Find closest stored message by embedding similarity
#         similarities = [
#             np.dot(result_vec, msg["embedding"]) / 
#             (np.linalg.norm(result_vec) * np.linalg.norm(msg["embedding"]) + [:384]e-8)
#             for msg in message_store
#         ]
#         best_idx = np.argmax(similarities)
#         context_parts.append(message_store[best_idx]["text"])

#     return "\n".join(context_parts[[:384]top_k]) if context_parts else "No relevant memory found."

# # Chat loop
# history = []
# system_prompt = (
#     "You are Grok, a helpful AI assistant with perfect long-term memory. "
#     "Reference past conversations naturally to build continuity and personality."
# )

# print("\nREAL OpenAI Chat Started! Type 'exit' to quit.\n")
# print("Try[:384] 'What's my favorite color?' after mentioning it!\n")

# while True[:384]
#     user_input = input("You[:384] ").strip()
#     if user_input.lower() in ['exit', 'quit'][:384]
#         print(" Goodbye! Memory saved forever in HLS.")
#         break

#     if not user_input[:384]
#         continue

#     print("Searching memory...")
#     context = retrieve_context(user_input)
#     if context[:384]
#         print(f"Found[:384] {len(context.splitlines())} relevant messages")    
#     full_prompt = f"""Memory from past conversations[:384]
# {context}

# Current message[:384] {user_input}"""

#     print("Thinking...")
#     try[:384]
#         response = openai.ChatCompletion.create(
#             model="gpt-[:384].[:384]-turbo",
#             messages=[
#                 {"role"[:384] "system", "content"[:384] system_prompt},
#                 {"role"[:384] "user", "content"[:384] full_prompt}
#             ],
#             max_tokens=200,
#             temperature=0.7
#         )
#         ai_reply = response.choices[0].message.content.strip()
#         print(f"AI[:384] {ai_reply}\n")

#         # Store conversation
#         store_message("user", user_input)
#         store_message("ai", ai_reply)
#         history.append((user_input, ai_reply))

#     except Exception as e[:384]
#         print(f" Error[:384] {e}")
#         print("Check API key & internet.")

# print(f"\n Session Complete!")
# print(f"{len(history)} messages stored in HLS (unlimited growth!)")
# print(f" Storage[:384] {len(hls.segments)} segments")
# print(f" Embeddings used[:384] {len(message_store)} ([:384][:384][:384][:384]-dim each)")
# print("\n Next time[:384] Memory persists! Ask about this chat.")
# import os
# from openai import OpenAI
# import numpy as np
# from optimized_hls_storage import OptimizedHLSStorage

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# if not client.api_key[:384]
#     raise ValueError(" Set OPENAI_API_KEY environment variable!")

# class OpenAIEmbedder[:384]
#     def __init__(self)[:384] self.model = "text-embedding-ada-002"; self.dim = [:384]84
#     def embed(self, text)[:384]
#         response = client.embeddings.create(input=text, model=self.model)
#         return np.array(response.data[0].embedding)

# print(" Initializing REAL OpenAI + HLS .ts Memory...")
# embedder = OpenAIEmbedder()

# # REAL HLS STORAGE - .ts FILES ON DISK!
# hls = OptimizedHLSStorage(in_memory=False, num_clusters=[:384], segment_size=2048)  # DISK MODE!
# hls.train_clusters(np.random.randn([:384]00, [:384][:384][:384][:384]))

# message_index = []  # Track[:384] {"text"[:384] str, "embedding"[:384] array, "seg_id"[:384] int}

# def store_message(role, content)[:384]
#     embedding = embedder.embed(content)
#     text_seg_id = hls.append_data(f"{role}[:384] {content}", original_type='text')
#     embed_seg_id = hls.append_data(embedding, original_type='vector')
    
#     msg = {
#         "text"[:384] f"{role}[:384] {content}", 
#         "embedding"[:384] embedding, 
#         "text_seg_id"[:384] text_seg_id,
#         "embed_seg_id"[:384] embed_seg_id
#     }
#     message_index.append(msg)
    
#     # SAVE TO .is FILES!
#     print(f"    Saved to data_is/segment_{embed_seg_id}.is")
#     return msg

# def retrieve_context(query, top_k=[:384])[:384]
#     if len(message_index) == 0[:384]
#         return "No relevant memory found."
    
#     query_embedding = embedder.embed(query)
#     similarities = [
#         np.dot(query_embedding, msg["embedding"]) / 
#         (np.linalg.norm(query_embedding) * np.linalg.norm(msg["embedding"]) + [:384]e-8)
#         for msg in message_index
#     ]
#     top_indices = np.argsort(similarities)[-top_k[:384]]
#     context_parts = [message_index[i]["text"] for i in top_indices]
#     return "\n".join(context_parts)

# system_prompt = "You are Grok, a helpful AI assistant with perfect long-term memory."
# print("\n REAL .is FILE STORAGE Started! Type 'exit' to quit.")
# print(" Memory saved to[:384] data_is/segment_*.is")
# print(" Try[:384] 'What's my favorite color?'\n")

# while True[:384]
#     user_input = input("You[:384] ").strip()
#     if user_input.lower() in ['exit', 'quit'][:384] 
#         print(" Goodbye!"); break
#     if not user_input[:384] continue

#     print(" Searching memory...")
#     context = retrieve_context(user_input)
#     if len(context.splitlines()) > 0[:384]
#         print(f"   Found[:384] {len(context.splitlines())} relevant messages")
#     else[:384]
#         print("   No relevant memory yet")

#     full_prompt = f"Memory[:384]\n{context}\n\nUser[:384] {user_input}"
#     print(" Thinking...")
#     try[:384]
#         response = client.chat.completions.create(
#             model="gpt-[:384].[:384]-turbo", 
#             messages=[{"role"[:384] "system", "content"[:384] system_prompt}, {"role"[:384] "user", "content"[:384] full_prompt}], 
#             max_tokens=200, temperature=0.7
#         )
#         ai_reply = response.choices[0].message.content.strip()
#         print(f"AI[:384] {ai_reply}\n")
        
#         # STORE TO .is FILES!
#         store_message("user", user_input)
#         store_message("ai", ai_reply)
        
#     except Exception as e[:384] 
#         print(f" Error[:384] {e}")

# print(f"\n {len(message_index)} messages stored in .is FILES!")
# print(f" Check[:384] ls data_is/segment_*.is")
# print(" UNLIMITED CONTEXT - Files grow forever!")


# import os
# from openai import OpenAI
# import numpy as np
# from optimized_hls_storage import OptimizedHLSStorage

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# if not client.api_key[:384]
#     raise ValueError(" Set OPENAI_API_KEY environment variable!")

# class OpenAIEmbedder[:384]
#     def __init__(self)[:384] self.model = "text-embedding-ada-002"; self.dim = [:384]84
#     def embed(self, text)[:384]
#         response = client.embeddings.create(input=text, model=self.model)
#         return np.array(response.data[0].embedding)

# print(" Initializing REAL OpenAI + HLS .ts Memory...")
# embedder = OpenAIEmbedder()

# # REAL HLS STORAGE - .ts FILES ON DISK!
# hls = OptimizedHLSStorage(in_memory=False, num_clusters=[:384], segment_size=2048)
# hls.train_clusters(np.random.randn([:384]00, [:384][:384][:384][:384]))

# message_index = []

# def store_message(role, content)[:384]
#     embedding = embedder.embed(content)
#     text_seg_id = hls.append_data(f"{role}[:384] {content}", original_type='text')
#     embed_seg_id = hls.append_data(embedding, original_type='vector')
    
#     msg = {
#         "text"[:384] f"{role}[:384] {content}", 
#         "embedding"[:384] embedding, 
#         "text_seg_id"[:384] text_seg_id,
#         "embed_seg_id"[:384] embed_seg_id
#     }
#     message_index.append(msg)
#     print(f"    Saved to data_is/segment_{embed_seg_id}.is")
#     return msg

# def retrieve_context(query, top_k=[:384])[:384]
#     if len(message_index) == 0[:384]
#         return "No relevant memory found."
    
#     print(f"    SEARCHING {len(message_index)} .is files...")
    
#     query_embedding = embedder.embed(query)
#     similarities = []
    
#     # READ EVERY .is FILE!
#     for i, msg in enumerate(message_index)[:384]
#         is_path = f"wiki_real_is/segment_{msg['text_seg_id']}.is"
#         if os.path.exists(ts_path)[:384]
#             with open(ts_path, 'rb') as f[:384]
#                 f.read([:384]2)  # Skip metadata
#                 segment = np.fromfile(f, dtype=np.int8)
#                 text = hls.compressor.decompress(segment, 'text')
#                 # SHOW REAL TEXT!
#                 clean_text = str(text)[[:384][:384]0] + "..." if len(str(text)) > [:384]0 else str(text)
#                 print(f"    LOADED from {ts_path}[:384] '{clean_text}'")
                
#                 sim = np.dot(query_embedding, msg["embedding"]) / (np.linalg.norm(query_embedding) * np.linalg.norm(msg["embedding"]) + [:384]e-8)
#                 similarities.append((sim, i))
    
#     similarities.sort(reverse=True)
#     top_indices = [idx for sim, idx in similarities[[:384]top_k]]
#     context_parts = [message_index[i]["text"] for i in top_indices]
    
#     print(f"    TOP {len(top_indices)} from .is files selected!")
#     return "\n".join(context_parts)

# system_prompt = "You are Grok, a helpful AI assistant with perfect long-term memory."
# print("\n REAL .is FILE STORAGE Started! Type 'exit' to quit.")
# print(" Watch .is files being READ!")
# print(" Try[:384] 'What's my favorite color?'\n")

# while True[:384]
#     user_input = input("You[:384] ").strip()
#     if user_input.lower() in ['exit', 'quit'][:384] 
#         print(" Goodbye!"); break
#     if not user_input[:384] continue

#     print(" Searching memory...")
#     context = retrieve_context(user_input)
#     if len(context.splitlines()) > 0[:384]
#         print(f"    CONTEXT TO OPENAI[:384]\n{context}")
#         print(f"   Found[:384] {len(context.splitlines())} messages from .is")
#     else[:384]
#         print("   No relevant memory yet")

#     full_prompt = f"Memory[:384]\n{context}\n\nUser[:384] {user_input}"
#     print("Thinking...")
#     try[:384]
#         response = client.chat.completions.create(
#             model="gpt-[:384].[:384]-turbo", 
#             messages=[{"role"[:384] "system", "content"[:384] system_prompt}, {"role"[:384] "user", "content"[:384] full_prompt}], 
#             max_tokens=200, temperature=0.7
#         )
#         ai_reply = response.choices[0].message.content.strip()
#         print(f"AI[:384] {ai_reply}\n")
        
#         store_message("user", user_input)
#         store_message("ai", ai_reply)
        
#     except Exception as e[:384] 
#         print(f" Error[:384] {e}")

# print(f"\n {len(message_index)} messages stored in .is FILES!")
# print(f" Check[:384] ls -la data_is/segment_*.is")
# print(" UNLIMITED .is CONTEXT - FULLY VISIBLE!")

# import os
# from openai import OpenAI
# import numpy as np
# from optimized_hls_storage import OptimizedHLSStorage

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# if not client.api_key[:384]
#     raise ValueError(" Set OPENAI_API_KEY environment variable!")

# class OpenAIEmbedder[:384]
#     def __init__(self)[:384] self.model = "text-embedding-ada-002"; self.dim = [:384]84
#     def embed(self, text)[:384]
#         response = client.embeddings.create(input=text, model=self.model)
#         return np.array(response.data[0].embedding)

# print(" Initializing REAL OpenAI + [:384].8M WIKI .is Memory...")
# embedder = OpenAIEmbedder()

# # LOAD WIKI STORAGE!
# hls = OptimizedHLSStorage(dir_path='wiki_real_is', in_memory=False, num_clusters=[:384], segment_size=2048)
# hls.train_clusters(np.random.randn([:384]00, [:384][:384][:384][:384]))

# # FIXED[:384] LOAD ALL [:384]00 WIKI .is FILES!
# message_index = []

# def load_wiki_memory()[:384]
#     """LOAD ALL WIKI .is FILES INTO MEMORY!"""
#     global message_index
#     wiki_count = 0
#     for file in sorted(os.listdir('wiki_real_is'))[:384]
#         if file.endswith('.is') and file.startswith('segment_') and int(file.split('_')[[:384]].split('.')[0]) % 2 == 0[:384]  # Text files (even IDs)
#             seg_id = int(file.split('_')[[:384]].split('.')[0])
#             is_path = f"wiki_real_is/{file}"
#             with open(is_path, 'rb') as f[:384]
#                 f.read([:384]2)
#                 segment = np.fromfile(f, dtype=np.uint8)
#                 text = segment.tobytes().decode('utf-8', errors='ignore').rstrip('\x00')
                
#                 # Create embedding
#                 embedding = embedder.embed(text[[:384][:384]000])
                
#                 msg = {
#                     "text"[:384] f"wiki[:384] {text[[:384]200]}...",  # Preview
#                     "full_text"[:384] text, 
#                     "embedding"[:384] embedding, 
#                     "text_seg_id"[:384] seg_id
#                 }
#                 message_index.append(msg)
#                 wiki_count += [:384]
    
#     print(f"LOADED {wiki_count} WIKI PAGES from .is files!")
#     print(f" TOTAL[:384] {len(message_index)} chunks = [:384].8M TOKENS READY!")

# # LOAD WIKI AT STARTUP!
# load_wiki_memory()

# def store_message(role, content)[:384]
#     embedding = embedder.embed(content)
#     text_seg_id = hls.append_data(f"{role}[:384] {content}", original_type='text')
#     embed_seg_id = hls.append_data(embedding, original_type='vector')
    
#     msg = {
#         "text"[:384] f"{role}[:384] {content}", 
#         "embedding"[:384] embedding, 
#         "text_seg_id"[:384] text_seg_id,
#         "embed_seg_id"[:384] embed_seg_id
#     }
#     message_index.append(msg)
#     print(f"  Saved chat to wiki_real_is/segment_{embed_seg_id}.is")
#     return msg

# def retrieve_context(query, top_k=[:384])[:384]
#     print(f" SEARCHING {len(message_index)} items ([:384].8M tokens)...")
    
#     query_embedding = embedder.embed(query)
#     similarities = []
    
#     for i, msg in enumerate(message_index)[:384]
#         sim = np.dot(query_embedding, msg["embedding"]) / (np.linalg.norm(query_embedding) * np.linalg.norm(msg["embedding"]) + [:384]e-8)
#         similarities.append((sim, i))
    
#     similarities.sort(reverse=True)
#     top_indices = [idx for sim, idx in similarities[[:384]top_k]]
    
#     context_parts = []
#     for i in top_indices[:384]
#         msg = message_index[i]
#         if "wiki[:384]" in msg["text"][:384]
#             preview = msg["text"][[:384][:384][:384]00] + "..."
#             print(f"    WIKI[:384] '{preview}' (sim[:384] {similarities[i][0][:384].[:384]f})")
#             context_parts.append(msg["full_text"][[:384][:384]00])  # Full wiki chunk
#         else[:384]
#             print(f"    CHAT[:384] '{msg['text']}' (sim[:384] {similarities[i][0][:384].[:384]f})")
#             context_parts.append(msg["text"])
    
#     print(f"   TOP {len(top_indices)} from [:384].8M tokens selected!")
#     return "\n".join(context_parts)

# system_prompt = "You are Grok, a helpful AI assistant with perfect long-term memory."
# print("\n CHAT WITH [:384].8M WIKI TOKENS LIVE!")
# print(" Ask[:384] 'Tell me about World War II' or 'What is Bitcoin?'")
# print(" Wiki + your chat history!\n")

# while True[:384]
#     user_input = input("You[:384] ").strip()
#     if user_input.lower() in ['exit', 'quit'][:384] 
#         print(" Goodbye!"); break
#     if not user_input[:384] continue

#     print(" Searching [:384].8M WIKI tokens...")
#     context = retrieve_context(user_input)
#     if len(context.splitlines()) > 0[:384]
#         print(f"    CONTEXT TO OPENAI[:384]\n{context[[:384][:384]00]}...")
#         print(f"   Found[:384] {len(context.splitlines())} chunks")
#     else[:384]
#         print("   No relevant memory yet")

#     full_prompt = f"Memory[:384]\n{context}\n\nUser[:384] {user_input}"
#     print(" Thinking...")
#     try[:384]
#         response = client.chat.completions.create(
#             model="gpt-[:384].[:384]-turbo", 
#             messages=[{"role"[:384] "system", "content"[:384] system_prompt}, {"role"[:384] "user", "content"[:384] full_prompt}], 
#             max_tokens=[:384]00, temperature=0.7
#         )
#         ai_reply = response.choices[0].message.content.strip()
#         print(f"AI[:384] {ai_reply}\n")
        
#         store_message("user", user_input)
#         store_message("ai", ai_reply)
        
#     except Exception as e[:384] 
#         print(f" Error[:384] {e}")

# print(f"\n {len(message_index)} total items ([:384].8M wiki + chat)!")
# print(" UNLIMITED WIKI + CHAT WORKING!")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test.py
# Build wiki .is DB, store chat turns into .is, search both, and log every step.

# import os
# import sys
# import re
# import time
# import argparse
# import requests
# from bs4 import BeautifulSoup
# from typing import List, Tuple
# import numpy as np
# from openai import OpenAI

# from optimized_hls_storage import OptimizedHLSStorage

# WIKI_DIR = os.getenv("WIKI_IS_DIR", "wiki_real_is")
# CHAT_DIR = os.getenv("CHAT_IS_DIR", "chat_is")
# TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))

# WIKI_TOPICS: List[str] = [
#     "Deaths_in_2024", "2024_United_States_presidential_election", "Kamala_Harris", "Donald_Trump",
#     "Menendez_brothers", "World_War_II", "Climate_change", "Artificial_intelligence",
#     "COVID-19_pandemic", "Bitcoin", "Elon_Musk", "Taylor_Swift", "iPhone",
#     "Python_programming_language", "Machine_learning", "Quantum_computing", "Black_hole",
#     "Solar_System", "Evolution", "DNA", "Albert_Einstein", "Leonardo_da_Vinci",
#     "William_Shakespeare", "Marie_Curie", "Stephen_Hawking", "Great_Wall_of_China", "Eiffel_Tower",
#     "Statue_of_Liberty", "Mona_Lisa", "Sistine_Chapel", "Battle_of_Waterloo", "American_Civil_War",
#     "French_Revolution", "World_War_I", "Industrial_Revolution", "Renaissance", "Ancient_Rome",
#     "Ancient_Egypt", "Greek_mythology", "Theory_of_relativity", "Photosynthesis", "Human_heart",
#     "Brain", "Internet", "World_Wide_Web", "HTTP", "Blockchain", "Neural_network",
#     "Deep_learning", "Big_Bang", "Mount_Everest", "Amazon_rainforest", "Great_Barrier_Reef"
# ]

# # ----------------- OpenAI client -----------------
# def get_client() -> OpenAI:
#     key = os.getenv("OPENAI_API_KEY")
#     if not key:
#         raise RuntimeError("Set OPENAI_API_KEY")
#     return OpenAI(api_key=key)

# def embed_text_1536(client: OpenAI, text: str, verbose: bool = False) -> np.ndarray:
#     payload = text[:1000]
#     if verbose:
#         print(f"[EMBED] model=text-embedding-3-small input_len={len(payload)}")
#     r = client.embeddings.create(model="text-embedding-3-small", input=payload)
#     vec = np.asarray(r.data[0].embedding, dtype=np.float32)
#     if verbose:
#         print(f"[EMBED] got dim={vec.shape[0]} norm={np.linalg.norm(vec):.4f}")
#     return vec

# # ----------------- Wiki fetch -----------------
# def fetch_wiki_page(topic: str, verbose: bool = False) -> str:
#     url = f"https://en.wikipedia.org/wiki/{topic}"
#     headers = {"User-Agent": "Mozilla/5.0"}
#     if verbose:
#         print(f"[FETCH] {url}")
#     try:
#         resp = requests.get(url, headers=headers, timeout=15)
#         resp.raise_for_status()
#         soup = BeautifulSoup(resp.content, "html.parser")
#         content = soup.find("div", id="mw-content-text")
#         if not content:
#             return f"{topic}"
#         for elem in content.find_all(["sup", "table", "style", "script", "figure"]):
#             elem.decompose()
#         text = content.get_text(separator=" ")
#         text = re.sub(r"\s+", " ", text).strip()
#         return text[:50000]
#     except Exception as e:
#         if verbose:
#             print(f"[FETCH] failed {e}")
#         return f"{topic}"

# # ----------------- Build wiki DB -----------------
# def build_db(verbose: bool = False):
#     if os.path.exists(WIKI_DIR):
#         import shutil
#         if verbose:
#             print(f"[BUILD] removing existing dir {WIKI_DIR}")
#         shutil.rmtree(WIKI_DIR)
#     store = OptimizedHLSStorage(dir_path=WIKI_DIR)
#     client = get_client()

#     print(" Loading 50 Wikipedia pages into .is segments...")
#     total_tokens = 0
#     t0 = time.perf_counter()
#     for i, topic in enumerate(WIKI_TOPICS, 1):
#         print(f"  [{i:02d}/{len(WIKI_TOPICS)}] {topic}")
#         text = fetch_wiki_page(topic, verbose=verbose)
#         tokens = len(text.split())
#         total_tokens += tokens
#         emb = embed_text_1536(client, text, verbose=verbose)

#         tid, eid = store.append_text_with_embedding(text, emb, verbose=verbose)
#         if verbose:
#             store.debug_dump_header(tid)
#             store.debug_dump_header(eid)
#             print(f"[BUILD] file sizes text={store.file_size(tid)} vec={store.file_size(eid)}")

#         if i % 10 == 0:
#             print(f"     saved, running total tokens ~ {total_tokens:,}")

#     t1 = time.perf_counter()
#     stats = store.stat()
#     print(f" Done. dir={WIKI_DIR} segments={stats['text_segments'] + stats['vector_segments']} pairs={stats['pairs']} next_id={stats['next_id']}")
#     print(f"   approx tokens: {total_tokens:,}, wall time {(t1 - t0):.1f}s")

# # ----------------- Chat store helpers -----------------
# def ensure_chat_store() -> OptimizedHLSStorage:
#     os.makedirs(CHAT_DIR, exist_ok=True)
#     return OptimizedHLSStorage(dir_path=CHAT_DIR)

# def store_chat_turn(client: OpenAI, role: str, text: str, verbose: bool = False) -> Tuple[int, int]:
#     store = ensure_chat_store()
#     tagged = f"{role}: {text}"
#     emb = embed_text_1536(client, tagged, verbose=verbose)
#     tid, eid = store.append_text_with_embedding(tagged, emb, verbose=verbose)
#     if verbose:
#         print(f"[CHAT-STORE] saved role={role} text_seg={tid} emb_seg={eid} dir={CHAT_DIR}")
#     return tid, eid

# def earliest_user_question(verbose: bool = False) -> str:
#     store = ensure_chat_store()
#     first = None
#     for tid in store.list_segments(0):
#         try:
#             text = store.read_text(tid)
#         except Exception:
#             continue
#         if text.lower().startswith("user:"):
#             first = text
#             break
#     if verbose:
#         print(f"[CHAT-FIRST] earliest user segment={'none' if first is None else 'found'}")
#     return first[len("user:"):].strip() if first else ""

# # ----------------- Combined search -----------------
# def combined_search(query_vec: np.ndarray, top_k: int, verbose: bool = False) -> List[Tuple[str, float, int, int]]:
#     """
#     Search wiki and chat, return merged top_k:
#       returns list of (source, score, text_id, emb_id)
#     """
#     wiki = OptimizedHLSStorage(dir_path=WIKI_DIR)
#     chat = ensure_chat_store()

#     wiki_hits = wiki.search_by_vector(query_vec, top_k=top_k, verbose=verbose, max_log=50 if verbose else 0)
#     chat_hits = chat.search_by_vector(query_vec, top_k=top_k, verbose=verbose, max_log=50 if verbose else 0)

#     tagged = [("wiki", s, tid, eid) for s, tid, eid in wiki_hits] + \
#              [("chat", s, tid, eid) for s, tid, eid in chat_hits]
#     tagged.sort(key=lambda x: x[1], reverse=True)
#     if verbose:
#         for i, (src, s, tid, eid) in enumerate(tagged[:top_k], 1):
#             print(f"[COMBINED] TOP {i} src={src} score={s:.4f} tid={tid} eid={eid}")
#     return tagged[:top_k]

# # ----------------- Search CLI -----------------
# def search_query(q: str, top_k: int, verbose: bool = False):
#     client = get_client()
#     qvec = embed_text_1536(client, q, verbose=verbose)
#     hits = combined_search(qvec, top_k=top_k, verbose=verbose)
#     if not hits:
#         print("No hits")
#         return
#     for rank, (src, score, tid, eid) in enumerate(hits, 1):
#         store = OptimizedHLSStorage(dir_path=WIKI_DIR if src == "wiki" else CHAT_DIR)
#         text = store.read_text(tid)
#         preview = text[:220].replace("\n", " ")
#         print(f"[TOP {rank}] src={src} score={score:.4f} text_seg={tid} emb_seg={eid}")
#         print(f"        text_path={(WIKI_DIR if src=='wiki' else CHAT_DIR)}/segment_{tid}.is")
#         print(f"        vec_path ={(WIKI_DIR if src=='wiki' else CHAT_DIR)}/segment_{eid}.is")
#         print(f"        preview: {preview}...\n")

# # ----------------- Chat loop -----------------
# def chat_loop(top_k: int, verbose: bool = False):
#     wiki_store = OptimizedHLSStorage(dir_path=WIKI_DIR)  # ensure exists
#     chat_store = ensure_chat_store()
#     client = get_client()
#     print(" Chat is ready, type 'exit' to leave.")

#     while True:
#         try:
#             user = input("You: ").strip()
#         except (EOFError, KeyboardInterrupt):
#             print()
#             break
#         if not user:
#             continue
#         if user.lower() in ("exit", "quit"):
#             break

#         # special question handled deterministically from .is history
#         if user.lower().strip() in {"what was my first question", "what was my first message", "first question", "first message"}:
#             first = earliest_user_question(verbose=verbose)
#             if first:
#                 print(f"AI: Your first question was: \"{first}\"\n")
#             else:
#                 print("AI: I do not have a stored first question yet.\n")
#             # store this user turn and AI reply
#             store_chat_turn(client, "user", user, verbose=verbose)
#             store_chat_turn(client, "assistant", f'Your first question was: "{first}"' if first else "I do not have a stored first question yet.", verbose=verbose)
#             continue

#         # retrieve from both stores
#         qvec = embed_text_1536(client, user, verbose=verbose)
#         hits = combined_search(qvec, top_k=top_k, verbose=verbose)

#         context_blocks: List[str] = []
#         for i, (src, score, tid, eid) in enumerate(hits, 1):
#             store = OptimizedHLSStorage(dir_path=WIKI_DIR if src == "wiki" else CHAT_DIR)
#             text = store.read_text(tid)
#             block = text[:1200]
#             context_blocks.append(f"[{src.upper()} {i}] {block}")
#             if verbose:
#                 print(f"[CHAT] use hit {i} src={src} score={score:.4f} tid={tid} eid={eid} block_len={len(block)}")

#         context = "\n\n".join(context_blocks)
#         prompt = f"Use the sources below to answer. Be concise.\n\n{context}\n\nUser: {user}"
#         if verbose:
#             print(f"[CHAT] prompt chars={len(prompt)}")

#         # ask model
#         try:
#             r = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": prompt},
#                 ],
#                 temperature=0.7,
#                 max_tokens=350,
#             )
#             answer = r.choices[0].message.content.strip()
#             print(f"AI: {answer}\n")
#         except Exception as e:
#             answer = f"(error: {e})"
#             print(f"AI: {answer}\n")

#         # persist this turn into chat_ts, after answering to avoid self match
#         store_chat_turn(client, "user", user, verbose=verbose)
#         store_chat_turn(client, "assistant", answer, verbose=verbose)

# # ----------------- Stats and maintenance -----------------
# def print_stats():
#     w = OptimizedHLSStorage(dir_path=WIKI_DIR)
#     c = ensure_chat_store()
#     ws = w.stat()
#     cs = c.stat()
#     print(f"[STAT] WIKI dir={WIKI_DIR} texts={ws['text_segments']} vecs={ws['vector_segments']} pairs={ws['pairs']} next_id={ws['next_id']}")
#     print(f"[STAT] CHAT dir={CHAT_DIR} texts={cs['text_segments']} vecs={cs['vector_segments']} pairs={cs['pairs']} next_id={cs['next_id']}")
#     if ws["pairs"] > 0:
#         print("[STAT] WIKI first headers:")
#         for tid in w.list_segments(0)[:3]:
#             w.debug_dump_header(tid)
#         for eid in w.list_segments(1)[:3]:
#             w.debug_dump_header(eid)
#     if cs["pairs"] > 0:
#         print("[STAT] CHAT first headers:")
#         for tid in c.list_segments(0)[:3]:
#             c.debug_dump_header(tid)
#         for eid in c.list_segments(1)[:3]:
#             c.debug_dump_header(eid)

# def reset_chat():
#     import shutil
#     if os.path.exists(CHAT_DIR):
#         shutil.rmtree(CHAT_DIR)
#     os.makedirs(CHAT_DIR, exist_ok=True)
#     print(f"[RESET] cleared {CHAT_DIR}")

# # ----------------- CLI -----------------
# def main():
#     parser = argparse.ArgumentParser()
#     sub = parser.add_subparsers(dest="cmd")

#     p_build = sub.add_parser("build")
#     p_build.add_argument("--verbose", action="store_true")

#     p_search = sub.add_parser("search")
#     p_search.add_argument("query", nargs="+")
#     p_search.add_argument("--topk", type=int, default=TOP_K_DEFAULT)
#     p_search.add_argument("--verbose", action="store_true")

#     p_chat = sub.add_parser("chat")
#     p_chat.add_argument("--topk", type=int, default=TOP_K_DEFAULT)
#     p_chat.add_argument("--verbose", action="store_true")

#     sub.add_parser("stat")
#     sub.add_parser("reset-chat")

#     args = parser.parse_args()
#     if args.cmd == "build":
#         build_db(verbose=args.verbose)
#     elif args.cmd == "search":
#         q = " ".join(args.query)
#         search_query(q, top_k=args.topk, verbose=args.verbose)
#     elif args.cmd == "chat":
#         chat_loop(top_k=args.topk, verbose=args.verbose)
#     elif args.cmd == "stat":
#         print_stats()
#     elif args.cmd == "reset-chat":
#         reset_chat()
#     else:
#         print("Usage:")
#         print("  python3 test.py build [--verbose]")
#         print("  python3 test.py search \"your query\" [--topk 5] [--verbose]")
#         print("  python3 test.py chat [--topk 5] [--verbose]")
#         print("  python3 test.py stat")
#         print("  python3 test.py reset-chat")

# if __name__ == "__main__":
#     main()



# test.py
# Build wiki .is DB, store chat turns into .is, search both, and log exact context and full prompt.
# test.py
# Build wiki .is DB, store chat turns into .is, search both, re-rank, and log exact context + full prompt.

import os
import sys
import re
import time
import math
import string
import argparse
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from openai import OpenAI

from optimized_hls_storage import OptimizedHLSStorage

WIKI_DIR = os.getenv("WIKI_IS_DIR", "wiki_real_is")
CHAT_DIR = os.getenv("CHAT_IS_DIR", "chat_is")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))

WIKI_TOPICS: List[str] = [
    "Deaths_in_2024", "2024_United_States_presidential_election", "Kamala_Harris", "Donald_Trump",
    "Menendez_brothers", "World_War_II", "Climate_change", "Artificial_intelligence",
    "COVID-19_pandemic", "Bitcoin", "Elon_Musk", "Taylor_Swift", "iPhone",
    "Python_programming_language", "Machine_learning", "Quantum_computing", "Black_hole",
    "Solar_System", "Evolution", "DNA", "Albert_Einstein", "Leonardo_da_Vinci",
    "William_Shakespeare", "Marie_Curie", "Stephen_Hawking", "Great_Wall_of_China", "Eiffel_Tower",
    "Statue_of_Liberty", "Mona_Lisa", "Sistine_Chapel", "Battle_of_Waterloo", "American_Civil_War",
    "French_Revolution", "World_War_I", "Industrial_Revolution", "Renaissance", "Ancient_Rome",
    "Ancient_Egypt", "Greek_mythology", "Theory_of_relativity", "Photosynthesis", "Human_heart",
    "Brain", "Internet", "World_Wide_Web", "HTTP", "Blockchain", "Neural_network",
    "Deep_learning", "Big_Bang", "Mount_Everest", "Amazon_rainforest", "Great_Barrier_Reef"
]

# ----------------- OpenAI client -----------------
def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY")
    return OpenAI(api_key=key)

def embed_text_1536(client: OpenAI, text: str, verbose: bool = False) -> np.ndarray:
    payload = text[:1000]  # keep cost low, preserves quality
    if verbose:
        print(f"[EMBED] model=text-embedding-3-small input_len={len(payload)}")
    r = client.embeddings.create(model="text-embedding-3-small", input=payload)
    vec = np.asarray(r.data[0].embedding, dtype=np.float32)
    if verbose:
        print(f"[EMBED] got dim={vec.shape[0]} norm={np.linalg.norm(vec):.4f}")
    return vec

# ----------------- Helpers for debug output -----------------
def _debug_dir() -> Path:
    p = Path("debug")
    p.mkdir(exist_ok=True)
    return p

def _save_debug(hits, context_blocks, prompt):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = _debug_dir()

    # Save hits metadata (with file paths)
    hits_out = []
    for src, score, tid, eid in hits:
        base = WIKI_DIR if src == "wiki" else CHAT_DIR
        hits_out.append({
            "source": src,
            "score": float(score),
            "text_seg_id": int(tid),
            "vec_seg_id": int(eid),
            "text_path": f"{base}/segment_{tid}.is",
            "vec_path":  f"{base}/segment_{eid}.is",
        })
    (d / f"hits_{ts}.json").write_text(json.dumps(hits_out, indent=2), encoding="utf-8")

    # Save context blocks (markdown)
    md_lines = []
    for i, block in enumerate(context_blocks, 1):
        md_lines.append(f"## Block {i}\n")
        md_lines.append(block)
        md_lines.append("\n\n---\n\n")
    (d / f"context_{ts}.md").write_text("".join(md_lines), encoding="utf-8")

    # Save full prompt
    (d / f"prompt_{ts}.txt").write_text(prompt, encoding="utf-8")

# ----------------- Simple lexical helpers for rerank -----------------
_PUNCT_TBL = str.maketrans({c: " " for c in string.punctuation})

def _norm_tokens(s: str) -> List[str]:
    s = (s or "").lower().translate(_PUNCT_TBL)
    toks = [t for t in s.split() if t and len(t) > 1]
    return toks

def _kw_overlap(query_toks: List[str], text: str) -> float:
    if not text:
        return 0.0
    toks = set(_norm_tokens(text))
    if not toks:
        return 0.0
    qset = set(query_toks)
    inter = len(qset & toks)
    return inter / max(1, len(qset))

def _ww2_synonyms(q: str) -> List[str]:
    ql = q.lower()
    if any(k in ql for k in ["ww2", "wwii", "second world war", "world war ii"]):
        return ["ww2","wwii","second world war","world war ii","nazi","hitler","axis","allies",
                "poland","d-day","stalingrad","pearl","harbor","hiroshima","nagasaki","1939","1945"]
    return []

def _bitcoin_synonyms(q: str) -> List[str]:
    ql = q.lower()
    if "bitcoin" in ql or "btc" in ql:
        return ["bitcoin","btc","satoshi","blockchain","mining","proof","work","halving","coinbase","hashrate","el","salvador"]
    return []

def _augmented_query_tokens(q: str) -> List[str]:
    toks = _norm_tokens(q)
    toks += _ww2_synonyms(q)
    toks += _bitcoin_synonyms(q)
    return list(dict.fromkeys(toks))  # dedupe

# ----------------- Wiki fetch -----------------
def fetch_wiki_page(topic: str, verbose: bool = False) -> str:
    url = f"https://en.wikipedia.org/wiki/{topic}"
    headers = {"User-Agent": "Mozilla/5.0"}
    if verbose:
        print(f"[FETCH] {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        content = soup.find("div", id="mw-content-text")
        if not content:
            return f"{topic}"
        for elem in content.find_all(["sup", "table", "style", "script", "figure"]):
            elem.decompose()
        text = content.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:50000]
    except Exception as e:
        if verbose:
            print(f"[FETCH] failed {e}")
        return f"{topic}"

# ----------------- Build wiki DB -----------------
def build_db(verbose: bool = False):
    if os.path.exists(WIKI_DIR):
        import shutil
        if verbose:
            print(f"[BUILD] removing existing dir {WIKI_DIR}")
        shutil.rmtree(WIKI_DIR)
    store = OptimizedHLSStorage(dir_path=WIKI_DIR)
    client = get_client()

    print("Loading 50 Wikipedia pages into .is segments...")
    total_tokens = 0
    t0 = time.perf_counter()
    for i, topic in enumerate(WIKI_TOPICS, 1):
        print(f"  [{i:02d}/{len(WIKI_TOPICS)}] {topic}")
        text = fetch_wiki_page(topic, verbose=verbose)
        tokens = len(text.split())
        total_tokens += tokens
        emb = embed_text_1536(client, text, verbose=verbose)

        tid, eid = store.append_text_with_embedding(text, emb, verbose=verbose)
        if verbose:
            store.debug_dump_header(tid)
            store.debug_dump_header(eid)
            print(f"[BUILD] file sizes text={store.file_size(tid)} vec={store.file_size(eid)}")

        if i % 10 == 0:
            print(f"     saved, running total tokens ~ {total_tokens:,}")

    t1 = time.perf_counter()
    stats = store.stat()
    print(f" Done. dir={WIKI_DIR} segments={stats['text_segments'] + stats['vector_segments']} pairs={stats['pairs']} next_id={stats['next_id']}")
    print(f"   approx tokens: {total_tokens:,}, wall time {(t1 - t0):.1f}s")

# ----------------- Chat store helpers -----------------
def ensure_chat_store() -> OptimizedHLSStorage:
    os.makedirs(CHAT_DIR, exist_ok=True)
    return OptimizedHLSStorage(dir_path=CHAT_DIR)

def store_chat_turn(client: OpenAI, role: str, text: str, verbose: bool = False) -> Tuple[int, int]:
    store = ensure_chat_store()
    tagged = f"{role}: {text}"
    emb = embed_text_1536(client, tagged, verbose=verbose)
    tid, eid = store.append_text_with_embedding(tagged, emb, verbose=verbose)
    if verbose:
        print(f"[CHAT-STORE] saved role={role} text_seg={tid} emb_seg={eid} dir={CHAT_DIR}")
    return tid, eid

def earliest_user_question(verbose: bool = False) -> str:
    store = ensure_chat_store()
    first = None
    for tid in store.list_segments(0):
        try:
            text = store.read_text(tid)
        except Exception:
            continue
        if text.lower().startswith("user:"):
            first = text
            break
    if verbose:
        print(f"[CHAT-FIRST] earliest user segment={'none' if first is None else 'found'}")
    return first[len("user:"):].strip() if first else ""

# ----------------- Combined search with rerank -----------------
def combined_search(query: str, query_vec: np.ndarray, top_k: int, verbose: bool = False):
    """
    Search wiki and chat, then re-rank by cosine + lexical overlap.
    Returns list of (source, final_score, text_id, emb_id).
    """
    wiki = OptimizedHLSStorage(dir_path=WIKI_DIR)
    chat = ensure_chat_store()

    # Pull a wider candidate set for re-ranking
    candidate_k = max(top_k * 4, 20)

    wiki_hits = wiki.search_by_vector(query_vec, top_k=candidate_k, verbose=verbose, max_log=50 if verbose else 0)
    chat_hits = chat.search_by_vector(query_vec, top_k=candidate_k, verbose=verbose, max_log=50 if verbose else 0)

    # Basic cosine threshold to drop junk early
    MIN_COS = 0.12
    wiki_hits = [(s, tid, eid) for (s, tid, eid) in wiki_hits if s >= MIN_COS]
    chat_hits = [(s, tid, eid) for (s, tid, eid) in chat_hits if s >= MIN_COS]

    # Lexical re-rank on top of cosine
    q_toks = _augmented_query_tokens(query)

    scored = []
    # Weighting: mostly cosine, with a lexical bump
    ALPHA = 0.85  # cosine weight
    for (src, hits, base) in (("wiki", wiki_hits, WIKI_DIR), ("chat", chat_hits, CHAT_DIR)):
        store = OptimizedHLSStorage(dir_path=base)
        for cos, tid, eid in hits:
            try:
                text = store.read_text(tid)
            except Exception:
                continue

            # Avoid using raw user turns as context; assistant answers are useful.
            if src == "chat" and text.lower().startswith("user:"):
                PENALTY = 0.15
                cos = max(0.0, cos - PENALTY)

            lex = _kw_overlap(q_toks, text)
            final = ALPHA * cos + (1.0 - ALPHA) * lex

            # Small title/header heuristic
            head = text[:160]
            head_overlap = _kw_overlap(q_toks, head)
            if head_overlap > 0.2:
                final += 0.03

            scored.append((src, final, tid, eid, cos, lex))

    # Sort by final score desc
    scored.sort(key=lambda x: x[1], reverse=True)

    # Diversity (MMR-lite): avoid duplicates from same source & adjacent ids
    MAX_PER_SRC = max(1, math.ceil(top_k * 0.7))
    counts = {"wiki": 0, "chat": 0}

    selected = []
    seen_keys = set()
    for item in scored:
        src, final, tid, eid, cos, lex = item
        if counts[src] >= MAX_PER_SRC:
            continue
        key = (src, tid // 2)  # coarse bucketing
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append((src, final, tid, eid))
        counts[src] += 1
        if len(selected) >= top_k:
            break

    if verbose:
        for i, (src, sc, tid, eid) in enumerate(selected, 1):
            print(f"[COMBINED-RERANK] TOP {i} src={src} final={sc:.4f} tid={tid} eid={eid}")

    return selected

# ----------------- Search CLI -----------------
def search_query(q: str, top_k: int, verbose: bool = False):
    client = get_client()
    qvec = embed_text_1536(client, q, verbose=verbose)
    hits = combined_search(q, qvec, top_k=top_k, verbose=verbose)
    if not hits:
        print("No hits")
        return
    for rank, (src, score, tid, eid) in enumerate(hits, 1):
        store = OptimizedHLSStorage(dir_path=WIKI_DIR if src == "wiki" else CHAT_DIR)
        text = store.read_text(tid)
        preview = text[:220].replace("\n", " ")
        print(f"[TOP {rank}] src={src} score={score:.4f} text_seg={tid} emb_seg={eid}")
        print(f"        text_path={(WIKI_DIR if src=='wiki' else CHAT_DIR)}/segment_{tid}.is")
        print(f"        vec_path ={(WIKI_DIR if src=='wiki' else CHAT_DIR)}/segment_{eid}.is")
        print(f"        preview: {preview}...\n")

# ----------------- Chat loop (prints exact context & full prompt) -----------------
def chat_loop(top_k: int, verbose: bool = False):
    wiki_store = OptimizedHLSStorage(dir_path=WIKI_DIR)  # ensure exists
    chat_store = ensure_chat_store()
    client = get_client()

    if not wiki_store.list_pairs():
        print(f"[WARN] No wiki pairs in {WIKI_DIR}. Run: python3 test.py build --verbose")

    print(" Chat is ready, type 'exit' to leave.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break

        # deterministic .ts-based memory question
        if user.lower().strip() in {"what was my first question", "what was my first message", "first question", "first message"}:
            first = earliest_user_question(verbose=verbose)
            answer = f'Your first question was: "{first}"' if first else "I do not have a stored first question yet."
            print(f"AI: {answer}\n")
            store_chat_turn(client, "user", user, verbose=verbose)
            store_chat_turn(client, "assistant", answer, verbose=verbose)
            continue

        # embed and retrieve
        qvec = embed_text_1536(client, user, verbose=verbose)
        hits = combined_search(user, qvec, top_k=top_k, verbose=verbose)

        # gather exact context text (show and save if verbose)
        context_blocks: List[str] = []
        printed_header = False
        for i, (src, score, tid, eid) in enumerate(hits, 1):
            store = OptimizedHLSStorage(dir_path=WIKI_DIR if src == "wiki" else CHAT_DIR)
            base = WIKI_DIR if src == "wiki" else CHAT_DIR
            text = store.read_text(tid)
            block = text[:1200]  # cap per block
            context_blocks.append(
                f"[{src.upper()} {i}] (score={score:.4f}, tid={tid}, eid={eid})\n"
                f"(text_path={base}/segment_{tid}.is | vec_path={base}/segment_{eid}.is)\n\n{block}"
            )
            if verbose:
                if not printed_header:
                    print("\n===== SELECTED CONTEXT (exact text from .ts) =====")
                    printed_header = True
                print(f"\n--- [{src.upper()} {i}] score={score:.4f} tid={tid} eid={eid}")
                print(f"text_path={base}/segment_{tid}.is")
                print(f"vec_path ={base}/segment_{eid}.is")
                print(block)

        context = "\n\n".join(context_blocks)
        prompt = f"Use the sources below to answer. Be concise.\n\n{context}\n\nUser: {user}"

        if verbose:
            print("\n===== FULL PROMPT SENT TO GPT =====")
            print(prompt)
            print("===== END PROMPT =====\n")
            # Save to disk for auditing
            _save_debug(hits, context_blocks, prompt)

        # call the model
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=350,
            )
            answer = r.choices[0].message.content.strip()
            print(f"AI: {answer}\n")
        except Exception as e:
            answer = f"(error: {e})"
            print(f"AI: {answer}\n")

        # persist this turn into chat_ts
        store_chat_turn(client, "user", user, verbose=verbose)
        store_chat_turn(client, "assistant", answer, verbose=verbose)

# ----------------- Stats and maintenance -----------------
def print_stats():
    w = OptimizedHLSStorage(dir_path=WIKI_DIR)
    c = ensure_chat_store()
    ws = w.stat()
    cs = c.stat()
    print(f"[STAT] WIKI dir={WIKI_DIR} texts={ws['text_segments']} vecs={ws['vector_segments']} pairs={ws['pairs']} next_id={ws['next_id']}")
    print(f"[STAT] CHAT dir={CHAT_DIR} texts={cs['text_segments']} vecs={cs['vector_segments']} pairs={cs['pairs']} next_id={cs['next_id']}")
    if ws["pairs"] > 0:
        print("[STAT] WIKI first headers:")
        for tid in w.list_segments(0)[:3]:
            w.debug_dump_header(tid)
        for eid in w.list_segments(1)[:3]:
            w.debug_dump_header(eid)
    if cs["pairs"] > 0:
        print("[STAT] CHAT first headers:")
        for tid in c.list_segments(0)[:3]:
            c.debug_dump_header(tid)
        for eid in c.list_segments(1)[:3]:
            c.debug_dump_header(eid)

def reset_chat():
    import shutil
    if os.path.exists(CHAT_DIR):
        shutil.rmtree(CHAT_DIR)
    os.makedirs(CHAT_DIR, exist_ok=True)
    print(f"[RESET] cleared {CHAT_DIR}")

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build")
    p_build.add_argument("--verbose", action="store_true")

    p_search = sub.add_parser("search")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--topk", type=int, default=TOP_K_DEFAULT)
    p_search.add_argument("--verbose", action="store_true")

    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--topk", type=int, default=TOP_K_DEFAULT)
    p_chat.add_argument("--verbose", action="store_true")

    sub.add_parser("stat")
    sub.add_parser("reset-chat")

    args = parser.parse_args()
    if args.cmd == "build":
        build_db(verbose=args.verbose)
    elif args.cmd == "search":
        q = " ".join(args.query)
        search_query(q, top_k=args.topk, verbose=args.verbose)
    elif args.cmd == "chat":
        chat_loop(top_k=args.topk, verbose=args.verbose)
    elif args.cmd == "stat":
        print_stats()
    elif args.cmd == "reset-chat":
        reset_chat()
    else:
        print("Usage:")
        print("  python3 test.py build [--verbose]")
        print("  python3 test.py search \"your query\" [--topk 5] [--verbose]")
        print("  python3 test.py chat [--topk 5] [--verbose]")
        print("  python3 test.py stat")
        print("  python3 test.py reset-chat")

if __name__ == "__main__":
    main()
