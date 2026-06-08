class DuplicateQuestionView {
  constructor() {
    this.form = document.querySelector("#question-form");
    this.question1 = document.querySelector("#question1");
    this.question2 = document.querySelector("#question2");
    this.clearButton = document.querySelector("#clear-button");
    this.predictButton = document.querySelector("#predict-button");
    this.statusPill = document.querySelector("#status-pill");
    this.resultBadge = document.querySelector("#result-badge");
    this.probabilityValue = document.querySelector("#probability-value");
    this.probabilityMeter = document.querySelector("#probability-meter");
    this.predictionValue = document.querySelector("#prediction-value");
    this.confidenceValue = document.querySelector("#confidence-value");
    this.cleanedOutput = document.querySelector("#cleaned-output");
    this.cleanedQuestion1 = document.querySelector("#cleaned-question1");
    this.cleanedQuestion2 = document.querySelector("#cleaned-question2");
  }

  bindSubmit(handler) {
    this.form.addEventListener("submit", (event) => {
      event.preventDefault();
      handler(this.question1.value.trim(), this.question2.value.trim());
    });
  }

  bindClear(handler) {
    this.clearButton.addEventListener("click", handler);
  }

  setLoading(isLoading) {
    this.predictButton.disabled = isLoading;
    this.predictButton.textContent = isLoading ? "Predicting" : "Predict";
    this.statusPill.textContent = isLoading ? "Working" : "Ready";
    this.statusPill.classList.toggle("is-loading", isLoading);
    this.statusPill.classList.remove("is-error");
  }

  showResult(result) {
    const probability = Number(result.duplicate_probability ?? 0);
    const confidence = Number(result.confidence ?? 0);
    const percentage = Math.round(probability * 100);

    this.resultBadge.textContent = result.label;
    this.resultBadge.classList.remove("is-error");
    this.probabilityValue.textContent = `${percentage}%`;
    this.probabilityMeter.style.width = `${percentage}%`;
    this.predictionValue.textContent = result.label;
    this.confidenceValue.textContent = result.confidence === null ? "--" : `${Math.round(confidence * 100)}%`;

    this.cleanedQuestion1.textContent = result.cleaned_questions.question1;
    this.cleanedQuestion2.textContent = result.cleaned_questions.question2;
    this.cleanedOutput.hidden = false;
  }

  showError(message) {
    this.statusPill.textContent = "Error";
    this.statusPill.classList.add("is-error");
    this.resultBadge.textContent = "Error";
    this.resultBadge.classList.add("is-error");
    this.predictionValue.textContent = message;
    this.confidenceValue.textContent = "--";
    this.probabilityValue.textContent = "--";
    this.probabilityMeter.style.width = "0%";
    this.cleanedOutput.hidden = true;
  }

  reset() {
    this.question1.value = "";
    this.question2.value = "";
    this.resultBadge.textContent = "Waiting";
    this.resultBadge.classList.remove("is-error");
    this.probabilityValue.textContent = "--";
    this.probabilityMeter.style.width = "0%";
    this.predictionValue.textContent = "Submit two questions";
    this.confidenceValue.textContent = "--";
    this.cleanedOutput.hidden = true;
    this.question1.focus();
  }
}
