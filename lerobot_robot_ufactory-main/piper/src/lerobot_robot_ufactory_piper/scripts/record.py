def main() -> None:
    # Importing our package (done before this module by Python) registers all
    # added config types. Reuse the parent project's recorder unchanged.
    from lerobot_robot_ufactory.scripts.uf_lerobot_record import main as parent_main

    parent_main()


if __name__ == "__main__":
    main()

