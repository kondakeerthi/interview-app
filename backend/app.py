#!/usr/bin/env python3
"""Simple Flask API exposing an equation solver."""

from __future__ import annotations

import os
import re
from flask import Flask, jsonify, request
from sympy import solve, Eq, simplify, Abs, S
from sympy.parsing.sympy_parser import parse_expr


def solve_equation(equation_str: str) -> str:
    """
    Solve a mathematical equation.
    
    Supports various formats:
    - "x^2 + 2x - 10" (implicitly equals 0)
    - "x + 5 = 10" (explicit equals)
    - "2x = 8"
    """
    # Normalize Unicode characters to ASCII equivalents
    # Replace Unicode minus (U+2212) with ASCII hyphen (U+002D)
    equation_str = equation_str.replace("−", "-")
    # Replace ^ with ** for Python exponentiation
    equation_str = equation_str.replace("^", "**")
    
    # Convert absolute value notation |expression| to Abs(expression)
    def replace_abs(match):
        return f"Abs({match.group(1)})"
    
    equation_str = re.sub(r'\|([^|]+)\|', replace_abs, equation_str)
    
    # Check if equation contains '=' sign
    if "=" in equation_str:
        # Split into left and right sides
        parts = equation_str.split("=", 1)
        if len(parts) != 2:
            raise ValueError("Invalid equation format")
        left_str, right_str = parts[0].strip(), parts[1].strip()
        
        # Parse both sides with implicit multiplication support
        try:
            left_expr = parse_expr(left_str, transformations='all')
            right_expr = parse_expr(right_str, transformations='all')
        except Exception as e:
            raise ValueError(f"Could not parse equation: {str(e)}")
        
        equation = Eq(left_expr, right_expr)
    else:
        try:
            expr = parse_expr(equation_str.strip(), transformations='all')
            equation = Eq(expr, 0)
        except Exception as e:
            raise ValueError(f"Could not parse expression: {str(e)}")
    
    # Find all symbols in the equation
    symbols_in_eq = equation.free_symbols
    
    if not symbols_in_eq:
        raise ValueError("No variable found in the equation. Please include a variable like 'x'.")
    
    if len(symbols_in_eq) > 1:
        # Multiple variables - solve for the first one (usually 'x')
        var = sorted(symbols_in_eq, key=str)[0]
    else:
        var = list(symbols_in_eq)[0]
    
    # Solve the equation
    try:
        abs_atoms = list(equation.atoms(Abs))
        if abs_atoms:
            # For absolute value equations: |expr| = c means expr = c or expr = -c
            if len(abs_atoms) == 1:
                abs_expr = abs_atoms[0]
                inner_expr = abs_expr.args[0]
                other_side = equation.rhs if abs_expr in equation.lhs.atoms(Abs) else equation.lhs
                
                if other_side.is_number:
                    eq1 = Eq(inner_expr, other_side)
                    eq2 = Eq(inner_expr, -other_side)
                    solutions = solve(eq1, var, dict=True) + solve(eq2, var, dict=True)
                else:
                    solutions = solve(equation, var, domain=S.Reals, dict=True)
            else:
                solutions = solve(equation, var, domain=S.Reals, dict=True)
        else:
            solutions = solve(equation, var, dict=True)
        
        if not solutions:
            return f"No solution found for {var}."
        
        def format_value(value):
            """Format a solution value with one decimal place."""
            value = simplify(value)
            return f"{value.evalf():.1f}" if value.is_number else str(value)
        
        if len(solutions) == 1:
            return f'{var} = "{format_value(solutions[0][var])}"'
        else:
            return ', '.join(f'{var} = "{format_value(sol[var])}"' for sol in solutions)
    
    except Exception as e:
        raise ValueError(f"Could not solve equation: {str(e)}")


def create_app() -> Flask:
    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(response):  # type: ignore[override]
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.get("/solve")
    def solve():
        equation = (request.args.get("equation") or "").strip()
        if not equation:
            return jsonify({"error": "Missing 'equation' query parameter"}), 400

        try:
            result = solve_equation(equation)
            return jsonify({"result": result})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Error solving equation: {str(e)}"}), 500

    @app.route("/", methods=["GET"])
    def root():
        return jsonify({"message": "Equation API. Try /solve?equation=1+1"})

    return app


def run() -> None:
    port = int(os.environ.get("PORT", 8000))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    run()
