"""ORM models matching diagrams/illustration/database_er.mmd."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Study(Base):
    __tablename__ = "study"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    publication_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    authors: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    datasets: Mapped[list["Dataset"]] = relationship(back_populates="study")


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    assay_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("study.id"), nullable=False)

    study: Mapped[Study] = relationship(back_populates="datasets")
    datasheets: Mapped[list["Datasheet"]] = relationship(back_populates="dataset")


class Scaffold(Base):
    __tablename__ = "scaffold"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sequence: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    datasheets: Mapped[list["Datasheet"]] = relationship(back_populates="scaffold")


class Datasheet(Base):
    """Catalog pointer to one exported CSV (``datasets/exported/…``).

    Per-edit rows are not stored in the database; they are loaded and filtered
    with Pandas behind the API.
    """

    __tablename__ = "datasheet"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)  # exported CSV
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), nullable=False)
    cell_line: Mapped[str] = mapped_column(String(64), nullable=False)
    pe_system: Mapped[str] = mapped_column(String(64), nullable=False)
    scaffold_id: Mapped[str] = mapped_column(ForeignKey("scaffold.id"), nullable=False)
    num_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    dataset: Mapped[Dataset] = relationship(back_populates="datasheets")
    scaffold: Mapped[Scaffold] = relationship(back_populates="datasheets")
