{
  description = "monpy development shell";

  inputs = {
    git-hooks-nix.url = "github:cachix/git-hooks.nix";
    git-hooks-nix.inputs.nixpkgs.follows = "nixpkgs";
    mohaus.url = "github:aarnphm/mohaus";
    mohaus.inputs.git-hooks-nix.follows = "git-hooks-nix";
    mohaus.inputs.nixpkgs.follows = "nixpkgs";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = {
    self,
    git-hooks-nix,
    mohaus,
    nixpkgs,
    ...
  }: let
    systems = [
      "aarch64-darwin"
      "x86_64-darwin"
      "aarch64-linux"
      "x86_64-linux"
    ];
    forAllSystems = fn:
      nixpkgs.lib.genAttrs systems (
        system:
          fn system (import nixpkgs {inherit system;})
      );

    commonPackages = pkgs: [
      pkgs.bash-completion
      pkgs.oxfmt
      pkgs.python311
      pkgs.uv
    ];

    mojoFormatHook = pkgs:
      pkgs.writeShellApplication {
        name = "monpy-mojo-format-check";
        runtimeInputs = [
          pkgs.coreutils
          pkgs.diffutils
        ];
        text = ''
          resolve_mojo() {
            if [ -n "''${MOHAUS_MOJO:-}" ] && [ -x "''${MOHAUS_MOJO:-}" ]; then
              printf '%s\n' "$MOHAUS_MOJO"
              return 0
            fi

            if command -v mojo >/dev/null 2>&1; then
              command -v mojo
              return 0
            fi

            if [ -n "''${MODULAR_DERIVED_PATH:-}" ] && [ -x "''${MODULAR_DERIVED_PATH:-}/build/bin/mojo" ]; then
              printf '%s\n' "$MODULAR_DERIVED_PATH/build/bin/mojo"
              return 0
            fi

            if [ -n "''${MODULAR_HOME:-}" ] && [ -x "''${MODULAR_HOME:-}/bin/mojo" ]; then
              printf '%s\n' "$MODULAR_HOME/bin/mojo"
              return 0
            fi

            return 1
          }

          if [ "$#" -eq 0 ]; then
            exit 0
          fi

          if ! mojo="$(resolve_mojo)"; then
            printf '%s\n' "mojo format skipped: no executable found via MOHAUS_MOJO, PATH, MODULAR_DERIVED_PATH, or MODULAR_HOME" >&2
            exit 0
          fi

          tmp="$(mktemp -d)"
          trap 'rm -rf "$tmp"' EXIT

          failed=0
          index=0
          for source in "$@"; do
            if [ ! -f "$source" ]; then
              continue
            fi

            case "$source" in
              *.mojo) ;;
              *) continue ;;
            esac

            copy="$tmp/$index.mojo"
            cp "$source" "$copy"
            "$mojo" format --line-length 119 --quiet "$copy"
            if ! diff -u "$source" "$copy" >&2; then
              printf '%s\n' "mojo format check failed: $source" >&2
              failed=1
            fi
            index=$((index + 1))
          done

          if [ "$failed" -ne 0 ]; then
            printf '%s\n' "run: mojo format --line-length 119 <files>" >&2
          fi

          exit "$failed"
        '';
      };

    oxfmtMarkdownHook = pkgs:
      pkgs.writeShellApplication {
        name = "monpy-oxfmt-markdown-check";
        runtimeInputs = [
          pkgs.coreutils
          pkgs.oxfmt
        ];
        text = ''
          config="$(mktemp "$PWD/.oxfmtrc.XXXXXX.json")"
          trap 'rm -f "$config"' EXIT

          cat > "$config" <<'JSON'
          {
            "ignorePatterns": ["docs/research/einsum-contraction.md"]
          }
          JSON

          oxfmt \
            --config "$config" \
            --check \
            --no-error-on-unmatched-pattern \
            "$@"
        '';
      };
  in {
    devShells = forAllSystems (
      system: pkgs: let
        mohausPackage = mohaus.packages.${system}.default or null;
        mohausBin =
          if mohausPackage == null
          then "mohaus"
          else "${mohausPackage}/bin/mohaus";
        mohausPathPrefix = pkgs.lib.optionalString (mohausPackage != null) "${mohausPackage}/bin:";
        preCommit = self.checks.${system}.pre-commit;
      in {
        default = pkgs.mkShell {
          packages =
            (commonPackages pkgs)
            ++ pkgs.lib.optional (mohausPackage != null) mohausPackage
            ++ preCommit.enabledPackages;

          shellHook = ''
            export UV_PYTHON="${pkgs.python311}/bin/python3.11"
            export UV_NO_MANAGED_PYTHON=1
            export UV_PYTHON_DOWNLOADS=never
            export PYTHON="$UV_PYTHON"
            export VIRTUAL_ENV="$PWD/.venv"

            if [ ! -x "$VIRTUAL_ENV/bin/python" ]; then
              uv venv --python "$UV_PYTHON" "$VIRTUAL_ENV"
            elif ! "$VIRTUAL_ENV/bin/python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))' >/dev/null 2>&1; then
              printf '%s\n' "recreating $VIRTUAL_ENV with Python 3.11 for monpy" >&2
              uv venv --python "$UV_PYTHON" --clear "$VIRTUAL_ENV"
            fi

            export PATH="${mohausPathPrefix}$VIRTUAL_ENV/bin:$PATH"

            if [ -z "''${MOHAUS_MOJO:-}" ] \
              && [ -n "''${MODULAR_DERIVED_PATH:-}" ] \
              && [ -x "''${MODULAR_DERIVED_PATH:-}/build/bin/mojo" ]; then
              export MOHAUS_MOJO="$MODULAR_DERIVED_PATH/build/bin/mojo"
            fi

            completion_root="$VIRTUAL_ENV/share"
            mkdir -p \
              "$completion_root/bash-completion/completions" \
              "$completion_root/fish/vendor_completions.d" \
              "$completion_root/zsh/site-functions"

            if command -v "${mohausBin}" >/dev/null 2>&1; then
              "${mohausBin}" completions bash > "$completion_root/bash-completion/completions/mohaus"
              "${mohausBin}" completions fish > "$completion_root/fish/vendor_completions.d/mohaus.fish"
              "${mohausBin}" completions zsh > "$completion_root/zsh/site-functions/_mohaus"
            else
              printf '%s\n' "mohaus completions skipped: mohaus is not available for ${system}" >&2
            fi

            export XDG_DATA_DIRS="$completion_root''${XDG_DATA_DIRS:+:$XDG_DATA_DIRS}"
            export FPATH="$completion_root/zsh/site-functions''${FPATH:+:$FPATH}"

            if [ -n "''${BASH_VERSION:-}" ] \
              && [ -f "$completion_root/bash-completion/completions/mohaus" ] \
              && type complete >/dev/null 2>&1; then
              . "$completion_root/bash-completion/completions/mohaus"
            fi

            if [ -n "''${ZSH_VERSION:-}" ]; then
              autoload -Uz compinit 2>/dev/null || true
              compinit -u 2>/dev/null || true
            fi

            if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
              ${preCommit.shellHook}
            fi
          '';
        };
      }
    );

    checks = forAllSystems (
      system: pkgs: {
        pre-commit = git-hooks-nix.lib.${system}.run {
          src = ./.;
          hooks = {
            mojo-format = {
              enable = true;
              name = "mojo format";
              entry = "${mojoFormatHook pkgs}/bin/monpy-mojo-format-check";
              files = "\\.mojo$";
            };

            oxfmt = {
              enable = true;
              name = "oxfmt markdown";
              entry = "${oxfmtMarkdownHook pkgs}/bin/monpy-oxfmt-markdown-check";
              files = "\\.(md|markdown|mdx)$";
              types_or = [];
            };

            alejandra = {
              enable = true;
              name = "alejandra";
              entry = "${pkgs.alejandra}/bin/alejandra --check";
              files = "\\.nix$";
            };

            deadnix = {
              enable = true;
              name = "deadnix";
              entry = "${pkgs.deadnix}/bin/deadnix --fail flake.nix";
              pass_filenames = false;
              files = "\\.nix$";
            };

            statix = {
              enable = true;
              name = "statix";
              entry = "${pkgs.statix}/bin/statix check flake.nix";
              pass_filenames = false;
              files = "\\.nix$";
            };

            check-added-large-files.enable = true;
            check-json.enable = true;
            check-merge-conflicts.enable = true;
            check-toml.enable = true;
            check-yaml.enable = true;
            trim-trailing-whitespace.enable = true;
          };
        };
      }
    );

    formatter = forAllSystems (_system: pkgs: pkgs.alejandra);
  };
}
