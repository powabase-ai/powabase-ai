"""Tool CRUD endpoints."""

import logging

from flask import Blueprint, jsonify, request

from ..auth import require_auth
from ..db import db
from ..models.tenant import Tool
from ..tools.builtin import BUILTIN_TOOL_DEFINITIONS

logger = logging.getLogger(__name__)

tools_bp = Blueprint("tools", __name__, url_prefix="/api/tools")


@tools_bp.route("", methods=["POST"])
@require_auth
def create_tool():
    data = request.get_json()
    name = data.get("name")
    description = data.get("description")
    tool_type = data.get("type")
    input_schema = data.get("input_schema")

    if not all([name, description, tool_type, input_schema]):
        return (
            jsonify({"error": "name, description, type, and input_schema are required"}),
            400,
        )

    tool = Tool(
        name=name,
        description=description,
        type=tool_type,
        input_schema=input_schema,
        config=data.get("config", {}),
    )
    db.session.add(tool)
    db.session.commit()

    return (
        jsonify(
            {
                "id": str(tool.id),
                "name": tool.name,
                "description": tool.description,
                "type": tool.type,
                "input_schema": tool.input_schema,
                "config": tool.config,
                "created_at": (tool.created_at.isoformat() if tool.created_at else None),
            }
        ),
        201,
    )


@tools_bp.route("", methods=["GET"])
@require_auth
def list_tools():
    builtins = [
        {
            "id": None,
            "name": d["name"],
            "description": d["description"],
            "type": "builtin",
            "input_schema": d["input_schema"],
            "config": {},
        }
        for d in BUILTIN_TOOL_DEFINITIONS
    ]
    custom_tools = Tool.query.all()
    customs = [
        {
            "id": str(t.id),
            "name": t.name,
            "description": t.description,
            "type": t.type,
            "input_schema": t.input_schema,
            "config": t.config,
            "created_at": (t.created_at.isoformat() if t.created_at else None),
        }
        for t in custom_tools
    ]
    return jsonify({"tools": builtins + customs})


@tools_bp.route("/<tool_id>", methods=["GET"])
@require_auth
def get_tool(tool_id: str):
    tool = db.session.get(Tool, tool_id)
    if not tool:
        return jsonify({"error": "Tool not found"}), 404
    return jsonify(
        {
            "id": str(tool.id),
            "name": tool.name,
            "description": tool.description,
            "type": tool.type,
            "input_schema": tool.input_schema,
            "config": tool.config,
        }
    )


@tools_bp.route("/<tool_id>", methods=["PUT"])
@require_auth
def update_tool(tool_id: str):
    tool = db.session.get(Tool, tool_id)
    if not tool:
        return jsonify({"error": "Tool not found"}), 404
    data = request.get_json()
    for field in ["name", "description", "input_schema", "config"]:
        if field in data:
            setattr(tool, field, data[field])
    db.session.commit()
    return jsonify({"id": str(tool.id), "name": tool.name})


@tools_bp.route("/<tool_id>", methods=["DELETE"])
@require_auth
def delete_tool(tool_id: str):
    tool = db.session.get(Tool, tool_id)
    if not tool:
        return jsonify({"error": "Tool not found"}), 404
    db.session.delete(tool)
    db.session.commit()
    return jsonify({"deleted": True})
