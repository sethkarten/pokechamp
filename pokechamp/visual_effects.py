#!/usr/bin/env python3
"""
Visual Effects Module - ASCII Art and Gradient Text for Pokemon Champion

Based on printing_guide.md techniques for creating engaging terminal output.
"""

import sys
from typing import Optional

try:
    import pyfiglet
    PYFIGLET_AVAILABLE = True
except ImportError:
    PYFIGLET_AVAILABLE = False

try:
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import fade
    FADE_AVAILABLE = True
except ImportError:
    FADE_AVAILABLE = False


class VisualEffects:
    """Visual effects manager for Pokemon Champion."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        
    def create_banner(self, text: str, font: str = "slant", style: str = "fire") -> str:
        """
        Create ASCII art banner with optional gradient.
        
        Args:
            text: Text to convert to ASCII art
            font: FIGlet font (slant, standard, block, doom)
            style: Gradient style (fire, water, greenblue, none)
            
        Returns:
            Formatted banner string
        """
        if not PYFIGLET_AVAILABLE:
            # Fallback: simple text formatting
            return f"\n{'='*50}\n  {text.upper()}\n{'='*50}\n"
            
        # Generate ASCII art
        ascii_art = pyfiglet.figlet_format(text, font=font)
        
        # Apply gradient if available
        if FADE_AVAILABLE and style != "none":
            gradient_map = {
                "fire": fade.fire,
                "water": fade.water,
                "greenblue": fade.greenblue,
                "purplepink": fade.purplepink,
                "brazil": fade.brazil,
            }
            
            gradient_func = gradient_map.get(style, fade.fire)
            return gradient_func(ascii_art)
        
        return ascii_art
    
    def battle_header(self, player1: str, player2: str) -> str:
        """Create a battle header with visual flair."""
        if RICH_AVAILABLE:
            # Use rich for colored output
            vs_text = f"{player1} VS {player2}"
            if PYFIGLET_AVAILABLE:
                ascii_vs = pyfiglet.figlet_format("BATTLE", font="standard")
                return f"{ascii_vs}\n{vs_text}\n" + "="*50
            else:
                return f"\n{'='*50}\n    BATTLE: {vs_text}\n{'='*50}\n"
        else:
            # Fallback ASCII
            return f"\n{'='*50}\n    {player1} VS {player2}\n{'='*50}\n"
    
    def system_status(self, component: str, status: str, color_style: str = "green") -> str:
        """Create styled status messages."""
        if RICH_AVAILABLE:
            status_map = {
                "success": "[green]✓[/green]",
                "error": "[red]✗[/red]", 
                "warning": "[yellow]⚠[/yellow]",
                "info": "[blue]ℹ[/blue]",
                "loading": "[cyan]⟳[/cyan]"
            }
            
            icon = status_map.get(status, "[white]•[/white]")
            return f"{icon} {component}"
        else:
            # Fallback text
            status_map = {
                "success": "[OK]",
                "error": "[ERROR]",
                "warning": "[WARN]",
                "info": "[INFO]", 
                "loading": "[LOAD]"
            }
            
            prefix = status_map.get(status, "[SYS]")
            return f"{prefix} {component}"
    
    def prediction_display(self, pokemon: str, predictions: list) -> str:
        """Display Pokemon predictions with visual formatting."""
        if not PYFIGLET_AVAILABLE:
            # Simple fallback
            lines = [f"\n=== PREDICTIONS FOR {pokemon.upper()} ==="]
            for item, prob in predictions[:5]:
                lines.append(f"  {item:<20} {prob:>6.1%}")
            return "\n".join(lines) + "\n"
        
        # ASCII art header for Pokemon name
        header = pyfiglet.figlet_format(pokemon[:8], font="small")  # Limit length
        
        if FADE_AVAILABLE:
            header = fade.greenblue(header)
        
        lines = [header, "BATTLE PREDICTIONS:"]
        for item, prob in predictions[:5]:
            bar_length = int(prob * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"  {item:<15} {bar} {prob:>6.1%}")
        
        return "\n".join(lines) + "\n"
    
    def minimax_progress(self, depth: int, nodes: int, time_taken: float) -> str:
        """Display minimax search progress."""
        if RICH_AVAILABLE:
            return f"[cyan]MINIMAX[/cyan] Depth: {depth} | Nodes: {nodes:,} | Time: {time_taken:.2f}s"
        else:
            return f"[MINIMAX] Depth: {depth} | Nodes: {nodes:,} | Time: {time_taken:.2f}s"
    
    def battle_turn(self, turn_number: int, action: str) -> str:
        """Format battle turn announcements."""
        if PYFIGLET_AVAILABLE and turn_number % 5 == 1:  # Every 5th turn gets ASCII
            turn_art = pyfiglet.figlet_format(f"T{turn_number}", font="small")
            if FADE_AVAILABLE:
                turn_art = fade.water(turn_art)
            return f"{turn_art}\nAction: {action}\n"
        else:
            return f"\n>>> TURN {turn_number}: {action}\n"
    
    def victory_banner(self, winner: str, turns: int) -> str:
        """Create victory celebration banner."""
        if PYFIGLET_AVAILABLE:
            victory_art = pyfiglet.figlet_format("VICTORY", font="banner")
            if FADE_AVAILABLE:
                victory_art = fade.brazil(victory_art)  # Gold gradient
            return f"{victory_art}\nWinner: {winner}\nTurns: {turns}\n"
        else:
            return f"\n{'='*50}\n    VICTORY: {winner}\n    Turns: {turns}\n{'='*50}\n"


# Global instance
visual = VisualEffects()


def print_banner(text: str, style: str = "fire"):
    """Convenience function to print a banner."""
    print(visual.create_banner(text, style=style))


def print_status(component: str, status: str):
    """Convenience function to print status."""
    if visual.console:
        visual.console.print(visual.system_status(component, status))
    else:
        print(visual.system_status(component, status))


def check_dependencies():
    """Check and report which visual libraries are available."""
    deps = {
        "pyfiglet": PYFIGLET_AVAILABLE,
        "rich": RICH_AVAILABLE, 
        "fade": FADE_AVAILABLE
    }
    
    print("\nVisual Effects Dependencies:")
    for lib, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {lib}")
    
    if not any(deps.values()):
        print("\nInstall visual libraries for enhanced output:")
        print("  pip install pyfiglet rich fade")
    
    return deps


if __name__ == "__main__":
    # Demo the visual effects
    check_dependencies()
    print_banner("POKEMON", "fire")
    print_status("Battle System", "success")
    print(visual.prediction_display("Pikachu", [("Thunder", 0.8), ("Quick Attack", 0.6)]))