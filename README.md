## 🔹 IDENTIFY-FAKE — Day 1 Documentation

### 🌍 WORLD (Understanding)

* Fake text = messages designed to deceive users
* Examples:

  * scam messages
  * fake offers
  * manipulation

---

### ⚙️ TECH (What I Built Today)

* Created dataset with:

  * `text` (input)
  * `label` (real/fake)
* Used:

  * `CountVectorizer` → converts text to numbers
  * `MultinomialNB` → simple AI model
* Model can now:

  * learn patterns
  * predict new text

---

### 🧠 PROBLEMS I FACED

* Error: `KeyError: 'text'`
* Cause: wrong column names (`Text`, `" Label"`)
* Fix:

  * cleaned dataset OR
  * used `.str.strip().str.lower()`

---

### 📈 WHAT I LEARNED

* Data must be clean
* Column names must match exactly
* AI depends on structured data
* Debugging is part of building

---

### 🚀 NEXT STEP

* Connect model to Flask
* Allow user input
* Show REAL / FAKE result

---

🔹 Day 2 — Flask Integration

⚙️ TECH
Saved trained model using pickle
Loaded model inside Flask
Created form for user input
Connected AI prediction to frontend



🧠 WHAT I LEARNED
AI models must be saved to reuse
Flask connects user input to backend logic
Prediction pipeline:
input → transform → predict → display

🔹 Day 3 — AI Confidence System

⚙️ TECH
Added predict_proba()
Calculated confidence score (%)
Displayed result + confidence in UI
Improved output readability

🧠 WHAT I LEARNED
AI does not just predict → it estimates probability
Confidence ≠ correctness
Data quality affects prediction strength

🔹 Day 4 — Data Intelligence Upgrade

⚙️ TECH
Expanded dataset with more real/fake examples
Retrained model
Improved prediction consistency

🌍 WORLD INSIGHT
Fake text uses urgency, manipulation, and unrealistic promises
Real text is calm, structured, and believable

🧠 WHAT I LEARNED
Data quality > model complexity
More examples = better learning
AI learns patterns, not truth
