import joblib

m = joblib.load("best_music_genre_v2.pkl")

print("Model type:", type(m))
print("Model steps:", m.named_steps)
print("Number of features:", m.n_features_in_)
print("C:", m.named_steps["model"].C)
print("Gamma:", m.named_steps["model"].gamma)