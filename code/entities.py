"""Typed domain model for the Buy or Wait? dataset."""

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path


class Currency(StrEnum):
    EUR = "EUR"
    IDR = "IDR"
    INR = "INR"
    USD = "USD"
    ZAR = "ZAR"


class RequestType(StrEnum):
    PURCHASE = "purchase"
    TRAVEL = "travel"
    EDUCATION = "education"
    FAMILY_TRANSFER = "family_transfer"
    DEBT_REPAYMENT = "debt_repayment"
    INVESTMENT = "investment"
    HOUSING = "housing"
    EMERGENCY_EXPENSE = "emergency_expense"
    OTHER = "other"


class EventType(StrEnum):
    EXPENSE = "expense"
    SUBSCRIPTION = "subscription"
    INCOME = "income"
    DEBT_PAYMENT = "debt_payment"
    REFUND = "refund"
    INVESTMENT_PURCHASE = "investment_purchase"
    INVESTMENT_SALE = "investment_sale"
    INVESTMENT_VALUATION = "investment_valuation"


class EventStatus(StrEnum):
    SETTLED = "settled"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNREALIZED = "unrealized"


class Direction(StrEnum):
    DEBIT = "debit"
    CREDIT = "credit"
    NON_CASH = "non_cash"


class Flexibility(StrEnum):
    FIXED = "fixed"
    REDUCIBLE = "reducible"
    STOPPABLE = "stoppable"
    REDUCIBLE_OR_STOPPABLE = "reducible_or_stoppable"


class PaymentMethod(StrEnum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"


class RecommendedMethod(StrEnum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class AffordabilityStatus(StrEnum):
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"


class MessageSource(StrEnum):
    EMPLOYER = "employer"
    SERVICE_PROVIDER = "service_provider"
    BANK = "bank"
    MERCHANT = "merchant"
    FINANCIAL_SERVICE = "financial_service"


class SpendingChangeKind(StrEnum):
    STOP = "stop"
    REDUCE_TO = "reduce_to"


@dataclass(frozen=True, slots=True)
class Profile:
    user_id: str
    home_currency: Currency
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    categories_to_protect: frozenset[str]
    categories_willing_to_reduce: frozenset[str]
    categories_willing_to_stop: frozenset[str]
    payment_methods: frozenset[PaymentMethod]
    max_installment_months: int | None

    def accepts(self, method: PaymentMethod) -> bool:
        return method in self.payment_methods


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    user_id: str
    event_type: EventType
    description: str
    category: str
    direction: Direction
    amount: Decimal | None
    currency: Currency
    event_date: dt.date
    settlement_date: dt.date | None
    status: EventStatus
    linked_event_id: str | None
    flexibility: Flexibility
    minimum_allowed_amount: Decimal | None

    @property
    def is_cash(self) -> bool:
        return self.direction is not Direction.NON_CASH

    @property
    def can_stop(self) -> bool:
        return self.flexibility in (Flexibility.STOPPABLE, Flexibility.REDUCIBLE_OR_STOPPABLE)

    @property
    def can_reduce(self) -> bool:
        return self.flexibility in (Flexibility.REDUCIBLE, Flexibility.REDUCIBLE_OR_STOPPABLE)


@dataclass(frozen=True, slots=True)
class Request:
    request_id: str
    user_id: str
    request_date: dt.date
    request_type: RequestType
    requested_amount: Decimal
    desired_completion_date: dt.date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True, slots=True)
class Payment:
    due_date: dt.date
    amount: Decimal


@dataclass(frozen=True, slots=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: PaymentMethod
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: dt.date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal

    def schedule(self) -> tuple[Payment, ...]:
        step = self.payment_frequency_days or 0
        return tuple(
            Payment(self.first_payment_date + dt.timedelta(days=step * i), self.payment_amount)
            for i in range(self.number_of_payments)
        )


@dataclass(frozen=True, slots=True)
class Message:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: dt.datetime
    source_type: MessageSource
    message_text: str


@dataclass(frozen=True, slots=True)
class ImageRef:
    image_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    path: Path


@dataclass(frozen=True, slots=True)
class SpendingChange:
    kind: SpendingChangeKind
    event_id: str
    new_amount: Decimal | None


@dataclass(frozen=True, slots=True)
class Decision:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: AffordabilityStatus
    recommended_payment_method: RecommendedMethod
    payment_plan: tuple[Payment, ...]
    earliest_date_for_full_payment: dt.date | None
    spending_changes: tuple[SpendingChange, ...]
    decision_explanation: str


@dataclass(frozen=True, slots=True)
class Dataset:
    profiles: dict[str, Profile]
    events: dict[str, Event]
    requests: dict[str, Request]
    sample_requests: dict[str, Request]
    sample_decisions: dict[str, Decision]
    payment_options: dict[str, tuple[PaymentOption, ...]]
    messages: dict[str, tuple[Message, ...]]
    images: dict[str, ImageRef]
    rates: dict[tuple[dt.date, Currency, Currency], Decimal]
    all_requests: dict[str, Request]
    events_by_user: dict[str, tuple[Event, ...]]
    requests_by_user: dict[str, tuple[Request, ...]]
    output_request_ids: tuple[str, ...]

    def rate(self, on: dt.date, src: Currency, dst: Currency) -> Decimal:
        if src == dst:
            return Decimal(1)
        return self.rates[(on, src, dst)]
