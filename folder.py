import os

# List of folder names to create
folders = [
"1. Linear Regression (Supervised Learning)",
"2. Logistic Regression (Supervised Learning)",
"3. Naive Bayes (Supervised Learning)",
"4. k-Nearest Neighbors (kNN) (Supervised Learning)",
"5. Decision Trees (Supervised Learning)",
"6. Support Vector Machines (SVM) (Supervised Learning)",
"7. Ridge Regression (Supervised Learning)",
"8. Lasso Regression (Supervised Learning)",
"9. Elastic Net Regression (Supervised Learning)",
"10. Support Vector Regression (SVR) (Supervised Learning)",
"11. Random Forest (Supervised Learning, Ensemble Method)",
"12. AdaBoost (Supervised Learning, Ensemble Method)",
"13. Gradient Boosting Machines (GBM) (Supervised Learning, Ensemble Method)",
"14. Extreme Gradient Boosting (XGBoost) (Supervised Learning, Ensemble Method)",
"15. LightGBM (Supervised Learning, Ensemble Method)",
"16. CatBoost (Supervised Learning, Ensemble Method)",
"17. Principal Component Analysis (PCA) (Unsupervised Learning, Dimensionality Reduction)",
"18. K-Means Clustering (Unsupervised Learning)",
"19. Hierarchical Clustering (Unsupervised Learning)",
"20. DBSCAN (Unsupervised Learning)",
"21. Affinity Propagation (Unsupervised Learning)",
"22. Spectral Clustering (Unsupervised Learning)",
"23. Apriori Algorithm (Associative Learning)",
"24. Eclat Algorithm (Associative Learning)",
"25. FP-Growth Algorithm (Associative Learning)",
"26. t-Distributed Stochastic Neighbor Embedding (t-SNE) (Unsupervised Learning, Dimensionality Reduction)",
"27. UMAP (Uniform Manifold Approximation and Projection) (Unsupervised Learning, Dimensionality Reduction)",
"28. Independent Component Analysis (ICA) (Unsupervised Learning, Dimensionality Reduction)",
"29. Self-Organizing Maps (SOMs) (Unsupervised Learning)",
"30. Q-Learning (Reinforcement Learning)",
"31. Temporal Difference (TD) Learning (Reinforcement Learning)",
"32. Deep Q-Network (DQN) (Reinforcement Learning)",
"33. Proximal Policy Optimization (PPO) (Reinforcement Learning)",
"34. Trust Region Policy Optimization (TRPO) (Reinforcement Learning)",
"35. Monte Carlo Tree Search (MCTS) (Reinforcement Learning)",
"36. Policy Gradient Methods (Reinforcement Learning)",
"37. Actor-Critic Methods (Reinforcement Learning)",
"38. Bayesian Networks (Probabilistic Models)",
"39. Hidden Markov Models (HMM) (Probabilistic Models)",
"40. Artificial Neural Networks (ANN) (Deep Learning)",
"41. Convolutional Neural Networks (CNN) (Deep Learning)",
"42. Recurrent Neural Networks (RNN) (Deep Learning)",
"43. Long Short-Term Memory Networks (LSTM) (Deep Learning)",
"44. Gated Recurrent Unit (GRU) (Deep Learning)",
"45. Autoencoders (Deep Learning, Unsupervised Learning)",
"46. Variational Autoencoders (VAE) (Deep Learning)",
"47. Generative Adversarial Networks (GANs) (Deep Learning)",
"48. U-Net (Deep Learning)",
"49. YOLO (You Only Look Once) (Deep Learning)",
"50. Siamese Networks (Deep Learning)",
"51. Transformer Networks (Deep Learning)",
"52. BERT (Bidirectional Encoder Representations from Transformers) (Deep Learning, NLP)",
"53. GPT (Generative Pre-trained Transformer) series (Deep Learning)",
"54. Deep Neural Networks (DNN) (Deep Learning)",
"55. Deep Reinforcement Learning Algorithms"
]

# Create each folder in the list
for folder in folders:
    # The folder will be created in the current working directory
    os.makedirs(folder, exist_ok=True)

"Folder(s) created successfully in the current working directory."

