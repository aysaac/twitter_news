from data_ingestion.transforms import register


@register(name="emoji2text")
def emoji2text(text: str) -> str:
    raise NotImplementedError()


@register(name="to_lower_case")
def to_lower_case(text: str) -> str:
    raise NotImplementedError()


@register(name="delete_stop_words")
def delete_stop_words(text: str) -> str:
    raise NotImplementedError()


@register(name="stem_words")
def stem_words(text: str) -> str:
    raise NotImplementedError()


@register(name="comjugate_stemed_words")
def cnojugate_stemed_words(text: str) -> str:
    raise NotImplementedError()

