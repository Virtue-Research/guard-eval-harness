# geh README demo recording — a vhs (https://github.com/charmbracelet/vhs) script.
#
# To regenerate `assets/demo.gif`:
#
#   brew install vhs            # macOS;  or `go install github.com/charmbracelet/vhs@latest`
#   cd /path/to/guard-eval-harness
#   vhs assets/demo.tape        # outputs assets/demo.gif
#
# Requires `geh` installed in PATH (`pip install geh` or `pip install -e .`)
# and a working internet connection for the first dataset fetch.

Output assets/demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 720
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 55ms
Set PlaybackSpeed 1.0

# --- Opening: minimal blank prompt so the GIF has a clean start frame
Hide
Type "clear" Enter
Type `export PS1='$ '` Enter
Type "clear" Enter
Show

Sleep 800ms

# --- Beat 1: discover what's available (proves "80+ datasets" is real)
Type "geh list packs"
Sleep 400ms
Enter
Sleep 2.5s

# --- Beat 2: run an entire benchmark pack against a model in one command
Type "geh run --pack core --model mock --limit 50"
Sleep 400ms
Enter
Sleep 3.5s

# --- Beat 3: export the results to a tidy table — the payoff
Type "geh export --run-dir out/packs/core-v1/mock --format csv --output /tmp/results.csv"
Sleep 400ms
Enter
Sleep 1.5s

Type "column -s, -t /tmp/results.csv | head"
Sleep 400ms
Enter
Sleep 3s

# Hold final frame so viewers can read the table
Sleep 1.5s
