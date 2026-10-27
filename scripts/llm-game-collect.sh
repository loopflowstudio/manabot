#!/bin/sh
# One archive retains complete decisions without thousands of object-store PUTs.
set -eu
record="$ETUDE_OUTPUT_DIR/../game-record"
if [ -d "$record" ]; then
    tar -czf "$ETUDE_OUTPUT_DIR/game.tar.gz.pending" -C "$record" .
    mv "$ETUDE_OUTPUT_DIR/game.tar.gz.pending" "$ETUDE_OUTPUT_DIR/game.tar.gz"
    for name in config summary progress; do
        if [ -f "$record/$name.json" ]; then
            cp "$record/$name.json" "$ETUDE_OUTPUT_DIR/$name.json"
        fi
    done
fi
