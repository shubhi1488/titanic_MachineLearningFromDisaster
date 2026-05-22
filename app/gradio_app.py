
import gradio as gr
import numpy as np
import joblib

# Load trained model
model = joblib.load('titanic_model.pkl')

# Prediction function
def predict_survival(
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked,
    title
):

    # Convert categorical values
    sex = 0 if sex == "Male" else 1

    embarked_mapping = {
        "Southampton": 0,
        "Cherbourg": 1,
        "Queenstown": 2
    }

    embarked = embarked_mapping[embarked]

    title_mapping = {
        "Mr": 1,
        "Miss": 2,
        "Mrs": 3,
        "Master": 4,
        "Rare": 5
    }

    title = title_mapping[title]

    # Feature engineering
    family_size = sibsp + parch + 1

    is_alone = 1 if family_size == 1 else 0

    # Final input
    features = np.array([[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked,
        family_size,
        is_alone,
        title
    ]])

    # Prediction
    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][prediction]

    # Result
    if prediction == 1:
        return f"Passenger Survived ✅ (Confidence: {probability:.2f})"

    else:
        return f"Passenger Did Not Survive ❌ (Confidence: {probability:.2f})"


# Gradio Interface
interface = gr.Interface(
    fn=predict_survival,

    inputs=[

        gr.Dropdown(
            [1, 2, 3],
            label="Passenger Class"
        ),

        gr.Radio(
            ["Male", "Female"],
            label="Gender"
        ),

        gr.Slider(
            0, 80,
            step=1,
            label="Age"
        ),

        gr.Slider(
            0, 10,
            step=1,
            label="Siblings/Spouse"
        ),

        gr.Slider(
            0, 10,
            step=1,
            label="Parents/Children"
        ),

        gr.Slider(
            0, 600,
            step=1,
            label="Fare"
        ),

        gr.Dropdown(
            ["Southampton",
             "Cherbourg",
             "Queenstown"],
            label="Embarked"
        ),

        gr.Dropdown(
            ["Mr",
             "Miss",
             "Mrs",
             "Master",
             "Rare"],
            label="Title"
        )
    ],

    outputs="text",

    title="Titanic Survival Prediction",

    description="""
    Enter passenger details to predict whether
    the passenger survived the Titanic disaster.
    """
)

# Launch app
interface.launch()
