def message(ex: BaseException) -> str:
    msg = str(ex)
    if msg:
        return msg
    return str(type(ex))
