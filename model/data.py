import dataclasses
from enum import IntEnum
from typing import Any, ClassVar, Dict, Optional, Type
from transformers import PreTrainedModel
from pydantic import BaseModel, Field, PositiveInt
import bittensor as bt

# The maximum bytes for metadata on the chain.
MAX_METADATA_BYTES = 128
# The length, in bytes, of a git commit hash.
GIT_COMMIT_LENGTH = 40
# The length, in bytes, of a base64 encoded sha256 hash.
SHA256_BASE_64_LENGTH = 44

class ModelIssue(Exception):
    '''
    Exception class to signal issues with models preventing evaluation.
    '''
    pass

class ModelLockedException(Exception):
    '''
    Exception class to signal a model is locked.
    '''
    pass

class ModelId(BaseModel):
    """Uniquely identifies a trained model"""

    # Makes the object "Immutable" once created.
    class Config:
        frozen = True
        extra = "forbid"

    MAX_REPO_ID_LENGTH: ClassVar[int] = (
        MAX_METADATA_BYTES - GIT_COMMIT_LENGTH - SHA256_BASE_64_LENGTH - 3  # separators
    )

    namespace: str = Field(
        description="Namespace where the model can be found. ex. Hugging Face username/org."
    )
    name: str = Field(
        description="Name of the model."
    )

    # When handling a model locally the commit and hash are not necessary.
    # Commit must be filled when trying to download from a remote store.
    commit: Optional[str] = Field(
        description="Commit of the model.",
        default=None
    )
    # Hash is filled automatically when uploading to or downloading from a remote store.
    hash: Optional[str] = Field(
        description="Hash of the trained model.",
        default=None
    )

    competition: Optional[str] = Field(
        description="Competition the model is submitted to.",
        default=None
    )

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return ':'.join([self.namespace,self.name,self.commit,self.hash,self.competition])

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["ModelId"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        keys = 'namespace','name','commit','hash','competition'
        if len(tokens) != len(keys):
            raise Exception(f'expecting {len(keys)} elements, found {len(tokens)}, in compressed string {cs}')
        return cls(**dict(zip(keys,tokens)))

    def format_label(self, full=False):
        c = ''
        if self.commit is not None:
            c = "|" + self.commit
            if not full:
                c = c[:5]
        return f"{self.namespace}/{self.name}{c}"

    @classmethod
    def dummy(cls, identifier: str) -> Type["ModelId"]:
        return cls(
            namespace='dummy',
            name=identifier,
        )

class Model(BaseModel):
    """Represents a pre trained foundation model."""

    class Config:
        arbitrary_types_allowed = True

    id: ModelId = Field(description="Identifier for this model.")
    # PreTrainedModel.base_model returns torch.nn.Module if needed.
    pt_model: PreTrainedModel = Field(description="Pre trained model.")


class ModelMetadata(BaseModel):
    id: ModelId = Field(description="Identifier for this trained model.")
    block: PositiveInt = Field(
        description="Block on which this model was claimed on the chain."
    )

    @staticmethod
    def parse_chain_data(metadata):
        model_id = None
        hex_data = None
        chain_str = None
        try:
            commitment = metadata["info"]["fields"][0]
            hex_data = commitment[list(commitment.keys())[0]][2:]
            chain_str = bytes.fromhex(hex_data).decode()
            model_id = ModelId.from_compressed_str(chain_str)
        except:
            # If the metadata format is not correct on the chain then we return None.
            bt.logging.trace(
                f"Failed to parse metadata {chain_str} / {hex_data} / {commitment}"
            )
            return None
        model_metadata = ModelMetadata(id=model_id, block=metadata["block"])
        return model_metadata
