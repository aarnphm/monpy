{
  description = "monpy development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      defaultModularRoot = "/Users/aarnphm/workspace/modular";
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python311
              pkgs.uv
            ];

            MOHAUS_MOJO = "${defaultModularRoot}/.derived/build/bin/mojo";
            MONPY_MOJO_KERNELS = "${defaultModularRoot}/Kernels";
            MONPY_MOJO_MAX_KERNELS = "${defaultModularRoot}/max/kernels/src";

            shellHook = ''
              export PATH="$PWD/.venv/bin:$PATH"
              echo "monpy shell: uv sync --extra dev"
              echo "mojo: $MOHAUS_MOJO"
            '';
          };
        }
      );
    };
}
