"""Load and validate the dataset/ CSVs into the typed model in entities.py."""

import csv
import datetime as dt
from decimal import Decimal, InvalidOperation
from pathlib import Path

from entities import (
    AffordabilityStatus,
    Currency,
    Dataset,
    Decision,
    Direction,
    Event,
    EventStatus,
    EventType,
    Flexibility,
    Flexibility as Flex,
    ImageRef,
    Message,
    MessageSource,
    Payment,
    PaymentMethod,
    PaymentOption,
    Profile,
    RecommendedMethod,
    Request,
    RequestType,
    SpendingChange,
    SpendingChangeKind,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
IMAGE_DIR = DATASET_DIR / "media" / "images"

REQUEST_COLUMNS = [
    "request_id",
    "user_id",
    "request_date",
    "request_type",
    "requested_amount",
    "desired_completion_date",
    "allows_partial_payment",
    "request_text",
]
DECISION_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]
EXPECTED_HEADERS = {
    "financial_profiles.csv": [
        "user_id",
        "home_currency",
        "current_available_balance",
        "minimum_balance_to_keep",
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
        "max_installment_months",
    ],
    "financial_events.csv": [
        "event_id",
        "user_id",
        "event_type",
        "description",
        "category",
        "direction",
        "amount",
        "currency",
        "event_date",
        "settlement_date",
        "status",
        "linked_event_id",
        "flexibility",
        "minimum_allowed_amount",
    ],
    "requests.csv": REQUEST_COLUMNS,
    "sample_requests.csv": REQUEST_COLUMNS + DECISION_COLUMNS[1:],
    "request_payment_options.csv": [
        "payment_option_id",
        "request_id",
        "payment_method",
        "payment_amount",
        "number_of_payments",
        "first_payment_date",
        "payment_frequency_days",
        "financing_fee",
        "total_payable_amount",
    ],
    "messages.csv": [
        "message_id",
        "user_id",
        "request_id",
        "related_event_id",
        "sent_at",
        "source_type",
        "message_text",
    ],
    "images.csv": ["image_id", "user_id", "request_id", "related_event_id"],
    "exchange_rates.csv": ["rate_date", "from_currency", "to_currency", "rate"],
    "output.csv": DECISION_COLUMNS,
}


class SchemaError(Exception):
    pass


class _Violations:
    def __init__(self) -> None:
        self.items: list[str] = []

    def check(self, ok: bool, message: str) -> None:
        if not ok:
            self.items.append(message)

    def raise_if_any(self) -> None:
        if not self.items:
            return
        shown = self.items[:25]
        more = len(self.items) - len(shown)
        tail = f"\n  ... and {more} more" if more > 0 else ""
        raise SchemaError(f"{len(self.items)} schema violation(s):\n  " + "\n  ".join(shown) + tail)


def _rows(name: str) -> list[dict[str, str]]:
    path = DATASET_DIR / name
    if not path.is_file():
        raise SchemaError(f"missing dataset file: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != EXPECTED_HEADERS[name]:
            raise SchemaError(
                f"{name}: header mismatch\n  expected {EXPECTED_HEADERS[name]}\n  found    {reader.fieldnames}"
            )
        return list(reader)


def _dec(value: str, where: str) -> Decimal:
    try:
        return Decimal(value.strip())
    except InvalidOperation:
        raise SchemaError(f"{where}: not a plain decimal: {value!r}") from None


def _opt_dec(value: str, where: str) -> Decimal | None:
    return _dec(value, where) if value.strip() else None


def _date(value: str, where: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value.strip())
    except ValueError:
        raise SchemaError(f"{where}: not an ISO date: {value!r}") from None


def _opt_date(value: str, where: str) -> dt.date | None:
    return _date(value, where) if value.strip() else None


def _opt_str(value: str) -> str | None:
    return value.strip() or None


def _bool(value: str, where: str) -> bool:
    match value.strip():
        case "true":
            return True
        case "false":
            return False
        case _:
            raise SchemaError(f"{where}: expected true/false, found {value!r}")


def _enum[E](enum_cls: type[E], value: str, where: str) -> E:
    try:
        return enum_cls(value.strip())
    except ValueError:
        raise SchemaError(f"{where}: {value!r} not a valid {enum_cls.__name__}") from None


def _pipe(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split("|") if part)


def _load_profiles() -> dict[str, Profile]:
    out: dict[str, Profile] = {}
    for row in _rows("financial_profiles.csv"):
        uid = row["user_id"]
        where = f"financial_profiles.csv[{uid}]"
        methods = frozenset(
            _enum(PaymentMethod, m, where) for m in _pipe(row["payment_methods_user_will_consider"])
        )
        months = row["max_installment_months"].strip()
        out[uid] = Profile(
            user_id=uid,
            home_currency=_enum(Currency, row["home_currency"], where),
            current_available_balance=_dec(row["current_available_balance"], where),
            minimum_balance_to_keep=_dec(row["minimum_balance_to_keep"], where),
            financial_priorities=_pipe(row["financial_priorities"]),
            categories_to_protect=frozenset(_pipe(row["expense_categories_to_protect"])),
            categories_willing_to_reduce=frozenset(
                _pipe(row["expense_categories_user_is_willing_to_reduce"])
            ),
            categories_willing_to_stop=frozenset(
                _pipe(row["expense_categories_user_is_willing_to_stop"])
            ),
            payment_methods=methods,
            max_installment_months=int(months) if months else None,
        )
    return out


def _load_events() -> dict[str, Event]:
    out: dict[str, Event] = {}
    for row in _rows("financial_events.csv"):
        eid = row["event_id"]
        where = f"financial_events.csv[{eid}]"
        out[eid] = Event(
            event_id=eid,
            user_id=row["user_id"],
            event_type=_enum(EventType, row["event_type"], where),
            description=row["description"],
            category=row["category"],
            direction=_enum(Direction, row["direction"], where),
            amount=_opt_dec(row["amount"], where),
            currency=_enum(Currency, row["currency"], where),
            event_date=_date(row["event_date"], where),
            settlement_date=_opt_date(row["settlement_date"], where),
            status=_enum(EventStatus, row["status"], where),
            linked_event_id=_opt_str(row["linked_event_id"]),
            flexibility=_enum(Flexibility, row["flexibility"], where),
            minimum_allowed_amount=_opt_dec(row["minimum_allowed_amount"], where),
        )
    return out


def _parse_request(row: dict[str, str], source: str) -> Request:
    rid = row["request_id"]
    where = f"{source}[{rid}]"
    return Request(
        request_id=rid,
        user_id=row["user_id"],
        request_date=_date(row["request_date"], where),
        request_type=_enum(RequestType, row["request_type"], where),
        requested_amount=_dec(row["requested_amount"], where),
        desired_completion_date=_date(row["desired_completion_date"], where),
        allows_partial_payment=_bool(row["allows_partial_payment"], where),
        request_text=row["request_text"],
    )


def _parse_plan(value: str, where: str) -> tuple[Payment, ...]:
    text = value.strip()
    if not text or text == "none":
        return ()
    payments = []
    for part in text.split("|"):
        day, _, amount = part.partition(":")
        if not amount:
            raise SchemaError(f"{where}: malformed payment_plan entry {part!r}")
        payments.append(Payment(_date(day, where), _dec(amount, where)))
    return tuple(payments)


def _parse_changes(value: str, where: str) -> tuple[SpendingChange, ...]:
    text = value.strip()
    if not text or text == "none":
        return ()
    changes = []
    for part in text.split("|"):
        fields = part.split(":")
        match fields:
            case ["stop", event_id]:
                changes.append(SpendingChange(SpendingChangeKind.STOP, event_id, None))
            case ["reduce_to", event_id, amount]:
                changes.append(
                    SpendingChange(SpendingChangeKind.REDUCE_TO, event_id, _dec(amount, where))
                )
            case _:
                raise SchemaError(f"{where}: malformed spending change {part!r}")
    return tuple(changes)


def _parse_decision(row: dict[str, str], source: str) -> Decision:
    rid = row["request_id"]
    where = f"{source}[{rid}]"
    return Decision(
        request_id=rid,
        amount_safe_to_pay=_dec(row["amount_safe_to_pay"], where),
        affordability_status=_enum(AffordabilityStatus, row["affordability_status"], where),
        recommended_payment_method=_enum(
            RecommendedMethod, row["recommended_payment_method"], where
        ),
        payment_plan=_parse_plan(row["payment_plan"], where),
        earliest_date_for_full_payment=_opt_date(row["earliest_date_for_full_payment"], where),
        spending_changes=_parse_changes(row["spending_changes_needed"], where),
        decision_explanation=row["decision_explanation"],
    )


def _load_payment_options() -> dict[str, tuple[PaymentOption, ...]]:
    grouped: dict[str, list[PaymentOption]] = {}
    for row in _rows("request_payment_options.csv"):
        oid = row["payment_option_id"]
        where = f"request_payment_options.csv[{oid}]"
        freq = row["payment_frequency_days"].strip()
        option = PaymentOption(
            payment_option_id=oid,
            request_id=row["request_id"],
            payment_method=_enum(PaymentMethod, row["payment_method"], where),
            payment_amount=_dec(row["payment_amount"], where),
            number_of_payments=int(row["number_of_payments"]),
            first_payment_date=_date(row["first_payment_date"], where),
            payment_frequency_days=int(freq) if freq else None,
            financing_fee=_dec(row["financing_fee"], where),
            total_payable_amount=_dec(row["total_payable_amount"], where),
        )
        grouped.setdefault(option.request_id, []).append(option)
    return {rid: tuple(sorted(v, key=lambda o: o.payment_option_id)) for rid, v in grouped.items()}


def _load_messages() -> dict[str, tuple[Message, ...]]:
    grouped: dict[str, list[Message]] = {}
    for row in _rows("messages.csv"):
        mid = row["message_id"]
        where = f"messages.csv[{mid}]"
        message = Message(
            message_id=mid,
            user_id=row["user_id"],
            request_id=_opt_str(row["request_id"]),
            related_event_id=_opt_str(row["related_event_id"]),
            sent_at=dt.datetime.fromisoformat(row["sent_at"].replace("Z", "+00:00")),
            source_type=_enum(MessageSource, row["source_type"], where),
            message_text=row["message_text"],
        )
        grouped.setdefault(message.user_id, []).append(message)
    return {uid: tuple(sorted(v, key=lambda m: m.sent_at)) for uid, v in grouped.items()}


def _load_images() -> dict[str, ImageRef]:
    out: dict[str, ImageRef] = {}
    for row in _rows("images.csv"):
        iid = row["image_id"]
        out[iid] = ImageRef(
            image_id=iid,
            user_id=row["user_id"],
            request_id=_opt_str(row["request_id"]),
            related_event_id=_opt_str(row["related_event_id"]),
            path=IMAGE_DIR / f"{iid}.png",
        )
    return out


def _load_rates() -> dict[tuple[dt.date, Currency, Currency], Decimal]:
    out: dict[tuple[dt.date, Currency, Currency], Decimal] = {}
    for row in _rows("exchange_rates.csv"):
        where = "exchange_rates.csv"
        key = (
            _date(row["rate_date"], where),
            _enum(Currency, row["from_currency"], where),
            _enum(Currency, row["to_currency"], where),
        )
        out[key] = _dec(row["rate"], where)
    return out


def validate(data: Dataset) -> None:
    v = _Violations()
    users = set(data.profiles)
    all_requests = data.all_requests

    for event in data.events.values():
        eid = event.event_id
        v.check(event.user_id in users, f"{eid}: unknown user {event.user_id}")
        v.check(
            (event.settlement_date is None) == (event.direction is Direction.NON_CASH),
            f"{eid}: settlement_date must be present iff the row is a cash movement",
        )
        if event.settlement_date is not None:
            v.check(
                event.settlement_date >= event.event_date,
                f"{eid}: settlement_date {event.settlement_date} precedes event_date {event.event_date}",
            )
        v.check(
            (event.minimum_allowed_amount is not None) == event.can_reduce,
            f"{eid}: minimum_allowed_amount must be present iff flexibility is reducible",
        )
        if event.linked_event_id is not None:
            v.check(
                event.linked_event_id in data.events,
                f"{eid}: linked_event_id {event.linked_event_id} not found",
            )

    for profile in data.profiles.values():
        v.check(
            (profile.max_installment_months is not None)
            == profile.accepts(PaymentMethod.INSTALLMENTS),
            f"{profile.user_id}: max_installment_months must be present iff installments are accepted",
        )

    for request in all_requests.values():
        rid = request.request_id
        v.check(request.user_id in users, f"{rid}: unknown user {request.user_id}")
        options = data.payment_options.get(rid, ())
        v.check(len(options) >= 2, f"{rid}: expected at least 2 payment options, found {len(options)}")
        full = [o for o in options if o.payment_method is PaymentMethod.FULL_PAYMENT]
        v.check(len(full) == 1, f"{rid}: expected exactly 1 full_payment option, found {len(full)}")
        for option in options:
            oid = option.payment_option_id
            v.check(
                option.payment_amount * option.number_of_payments == option.total_payable_amount,
                f"{oid}: payment_amount x number_of_payments != total_payable_amount",
            )
            v.check(
                request.requested_amount + option.financing_fee == option.total_payable_amount,
                f"{oid}: requested_amount + financing_fee != total_payable_amount",
            )
            v.check(
                (option.payment_frequency_days is None)
                == (option.payment_method is PaymentMethod.FULL_PAYMENT),
                f"{oid}: payment_frequency_days must be present iff the option is installments",
            )

    for rid in data.payment_options:
        v.check(rid in all_requests, f"payment options reference unknown request {rid}")

    for messages in data.messages.values():
        for message in messages:
            mid = message.message_id
            v.check(message.user_id in users, f"{mid}: unknown user {message.user_id}")
            v.check(
                message.request_id is None or message.request_id in all_requests,
                f"{mid}: unknown request {message.request_id}",
            )
            v.check(
                message.related_event_id is None or message.related_event_id in data.events,
                f"{mid}: unknown related_event_id {message.related_event_id}",
            )

    blank_amount = {e.event_id for e in data.events.values() if e.amount is None}
    image_targets = {i.related_event_id for i in data.images.values() if i.related_event_id}
    v.check(
        blank_amount == image_targets,
        f"blank-amount events {sorted(blank_amount - image_targets)} have no image; "
        f"images target non-blank events {sorted(image_targets - blank_amount)}",
    )
    for image in data.images.values():
        v.check(image.path.is_file(), f"{image.image_id}: missing file {image.path}")
        v.check(image.user_id in users, f"{image.image_id}: unknown user {image.user_id}")

    v.check(
        set(data.output_request_ids) == set(data.requests),
        "output.csv request_ids do not match requests.csv",
    )
    v.raise_if_any()


def load_dataset(*, run_validation: bool = True) -> Dataset:
    events = _load_events()
    events_by_user: dict[str, list[Event]] = {}
    for event in events.values():
        events_by_user.setdefault(event.user_id, []).append(event)

    sample_rows = _rows("sample_requests.csv")
    requests = {
        row["request_id"]: _parse_request(row, "requests.csv") for row in _rows("requests.csv")
    }
    sample_requests = {
        row["request_id"]: _parse_request(row, "sample_requests.csv") for row in sample_rows
    }
    all_requests = {**sample_requests, **requests}

    requests_by_user: dict[str, list[Request]] = {}
    for request in all_requests.values():
        requests_by_user.setdefault(request.user_id, []).append(request)

    data = Dataset(
        profiles=_load_profiles(),
        events=events,
        requests=requests,
        sample_requests=sample_requests,
        sample_decisions={
            row["request_id"]: _parse_decision(row, "sample_requests.csv") for row in sample_rows
        },
        payment_options=_load_payment_options(),
        messages=_load_messages(),
        images=_load_images(),
        rates=_load_rates(),
        all_requests=all_requests,
        events_by_user={
            uid: tuple(sorted(v, key=lambda e: (e.event_date, e.event_id)))
            for uid, v in events_by_user.items()
        },
        requests_by_user={
            uid: tuple(sorted(v, key=lambda r: (r.request_date, r.request_id)))
            for uid, v in requests_by_user.items()
        },
        output_request_ids=tuple(row["request_id"] for row in _rows("output.csv")),
    )
    if run_validation:
        validate(data)
    return data


if __name__ == "__main__":
    data = load_dataset()
    print(
        f"profiles={len(data.profiles)} events={len(data.events)} "
        f"requests={len(data.requests)} samples={len(data.sample_requests)} "
        f"options={sum(len(o) for o in data.payment_options.values())} "
        f"messages={sum(len(m) for m in data.messages.values())} "
        f"images={len(data.images)} rates={len(data.rates)}"
    )
