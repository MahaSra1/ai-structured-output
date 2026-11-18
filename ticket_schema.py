"""Day 11: the Pydantic "customs officer" for SwiftParcel tickets.

The LLM writes JSON. This model decides if it is allowed in:
- right fields, right types, allowed values (shape)
- tracking number must really appear in the customer's message (grounding)
"""
import re
from enum import Enum
from typing import Optional

from pydantic import (BaseModel, ConfigDict, Field, ValidationInfo,
                      field_validator, model_validator)

TRACKING_RE = re.compile(r"SP\d{4}")


class IssueType(str, Enum):
    DAMAGED = "damaged"
    LOST = "lost"
    LATE = "late"
    REFUND = "refund"
    ADDRESS_CHANGE = "address_change"
    UNCLEAR = "unclear"


class Ticket(BaseModel):
    # extra="forbid": the LLM cannot sneak in fields we didn't ask for
    model_config = ConfigDict(extra="forbid")

    # Field(...) = REQUIRED but can be null. A default (None) would let the LLM
    # skip the key, and Pydantic does NOT run validators on defaults.
    customer_name: Optional[str] = Field(
        ..., description="Customer's name if written in the message, else null")
    tracking_number: Optional[str] = Field(
        ..., description="SP followed by 4 digits (e.g. SP1002), only if written in the message, else null")
    issues: list[IssueType] = Field(
        ..., min_length=1, description="Every issue mentioned. Use ['unclear'] alone if there is no real issue")
    customer_upset: bool = Field(
        ..., description="true only if the customer sounds angry or frustrated")
    summary: str = Field(
        ..., min_length=5, max_length=200, description="One short sentence, facts only")

    # 1) BEFORE type checks: tidy up what small models typically return
    @field_validator("tracking_number", mode="before")
    @classmethod
    def tidy_tracking(cls, v):
        if v is None:
            return None
        v = str(v).strip().upper().replace(" ", "").replace("-", "")
        return None if v in {"", "NULL", "NONE", "N/A", "UNKNOWN"} else v

    # 2) AFTER type checks: format + grounding (needs the message as context)
    @field_validator("tracking_number")
    @classmethod
    def check_tracking(cls, v, info: ValidationInfo):
        message = (info.context or {}).get("message")
        clean_msg = message.upper().replace(" ", "").replace("-", "") if message else None

        if v is None:
            # Completeness: catches MISSED numbers (null is not OK if one is in the message)
            if clean_msg:
                found = TRACKING_RE.search(clean_msg)
                if found:
                    raise ValueError(f"message contains '{found.group()}' but you returned null. "
                                     "Copy it exactly")
            return v
        if not TRACKING_RE.fullmatch(v):
            raise ValueError(f"must look like SP1234, got '{v}'")
        # Grounding: catches INVENTED numbers
        if clean_msg is not None and v not in clean_msg:
            raise ValueError(f"'{v}' is not in the customer message (invented?). Use null")
        return v

    @field_validator("issues")
    @classmethod
    def dedupe_issues(cls, v):
        return list(dict.fromkeys(v))  # remove duplicates, keep order

    # 3) Whole-object rule: checks that need more than one field
    @model_validator(mode="after")
    def unclear_alone(self):
        if IssueType.UNCLEAR in self.issues and len(self.issues) > 1:
            raise ValueError("'unclear' cannot be combined with other issues")
        return self
