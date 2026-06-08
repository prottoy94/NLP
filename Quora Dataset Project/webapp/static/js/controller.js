class DuplicateQuestionController {
  constructor(api, view) {
    this.api = api;
    this.view = view;
    this.view.bindSubmit(this.handleSubmit.bind(this));
    this.view.bindClear(this.handleClear.bind(this));
  }

  async handleSubmit(question1, question2) {
    if (!question1 || !question2) {
      this.view.showError("Both questions are required.");
      return;
    }

    this.view.setLoading(true);

    try {
      const result = await this.api.predict(question1, question2);
      this.view.showResult(result);
    } catch (error) {
      this.view.showError(error.message);
    } finally {
      this.view.setLoading(false);
    }
  }

  handleClear() {
    this.view.reset();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const api = new DuplicateQuestionApi();
  const view = new DuplicateQuestionView();
  new DuplicateQuestionController(api, view);
});
