# LLM from Scratch

# Overview
This is a GPT model trained from scratch on the TinyStories dataset. Following Karpathy's NanoGPT tutorial, the goal of this project was to dive deeper into the inner workings of LLMs. 

Details
- tokenizer is character-level: it finds the set of all characters in the dataset
- model has 0.825M params:
  - context length: 64
  - batch size: 32
  - embedding size: 64
  - d_k and d_v == embedding size
  - 6 layers
  - 6 heads
  - dropout: 0.2
- trained for 5000 epochs, with a checkpoint (training state saved too) every 1000 epochs
- checkpoints saved as .pt (torch load) for simplicity

For a 0.825M model, that was trained for ~11min on a macbook, the results are surprising:
(limited to 2K output tokens)
```
. So excited, "Where it sand. They had the wind down. They wanted to go home. It was tried roll that on, he she had becore by for a big duch. She had a said, "Mom and Wen the too keap!" But said, she didn't know it came and the sun. He jot for the halpy on. It doods was a big big flace inside. She had not room would looked eather. The wanted to feight it and was very locts of his dall thrughth, it was was very rible. He saw anytoe things the little girl's happy that was a lift out important to it in the box. He hug to draw and encing calling and feel the little boy was book! Cit chiled with the mom on it too by tooks with play so friends." they both Mia back quiesting flew ug suite. Lily was shappy to her it to saap inside his seany, "I took not," started. You do your me we vat to go of about it. It was walking home." He gave to play outside in the box, she coor and started to feacher it was a doll. He was sturning so climbins to go have and higher, be's scared and kept and She started to burn it. Their box, "I did not wave a reach reach with his dor lead of the congs."
The boy waved to scelled when her favourl walking them. The wind bird looking around the box. He was if hole. I proud hard to the room and kitchen. She was ging. It had her could too, so that biked Lily. But she was happy and fun playing with his frel you. But he had started to seach nosise her adoifh it a big time. She asked his mone, she said. "I'm what it strong?" strock. He was so carped and Max better who liked to go to the hole clittle. She wouldn't still and sple.
<|endoftext|>

Once there was a little bad girl named Lily named Lily, "Look, at his mom like to play with a staople. I sorry box he looked at delicious” Mom secided becausifus! Tim that not it was very time to little girl named Satches, it believed the ground.
"No, careful, can colors and his saw, Jreal! The box's monkey about a beautiful of job to looks. The brown the stro off birdfelited the cuisquir careful in in the punoing and
```

# Extensions
I wrote my own tokenizer at character-level, just for the sake of doing it manually. Using an existing tokenizer like Sentencepiece (and training that) should yield massive improvements.


Pre-processing the dataset is really inefficient. It's a 1.9GB txt, and gets read into Python all at once, and then is tokenized and converted to tensors; this entire process used ~20GB of swap, on top of my 16GB RAM. It would be good to build a lazy dataloader... but in general, just don't train anything on 16GB ram...
