from trulens_eval.feedback.provider import OpenAI
from trulens_eval import Feedback, TruLlama
from trulens_eval.app import App
import numpy as np
from trulens_eval.feedback import Groundedness


class FeedbackManager:
    def __init__(self, query_engine):
        self.query_engine = query_engine
        self.provider = OpenAI()
        self.context = App.select_context(self.query_engine)
        self.setup_feedback()

    def setup_feedback(self):
        grounded = Groundedness(groundedness_provider=OpenAI())
        self.f_groundedness = (
            Feedback(grounded.groundedness_measure_with_cot_reasons)
            .on(self.context.collect())
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
        )

        self.f_answer_relevance = (
            Feedback(self.provider.relevance)
            .on_input_output()
        )

        self.f_context_relevance = (
            Feedback(self.provider.context_relevance_with_cot_reasons)
            .on_input()
            .on(self.context)
            .aggregate(np.mean)
        )

    def record_query(self, query):
        with TruLlama(self.query_engine, app_id='LlamaIndex_App1', feedbacks=[self.f_groundedness, self.f_answer_relevance, self.f_context_relevance]) as recording:
            self.query_engine.query(query)
        return recording.records
