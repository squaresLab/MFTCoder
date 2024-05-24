import os
import sys
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
import argparse


def load_model_tokenizer(
    path,
    model_type=None,
    peft_path=None,
    torch_dtype=torch.bfloat16,
    quantization=None,
    eos_token=None,
    pad_token=None,
):
    """
    load model and tokenizer by transfromers
    """

    # load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    config, unused_kwargs = AutoConfig.from_pretrained(
        path, trust_remote_code=True, return_unused_kwargs=True
    )
    print("unused_kwargs:", unused_kwargs)
    print("config input:\n", config)

    if eos_token:
        eos_token = eos_token
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        print(f"eos_token {eos_token} from user input")
    else:
        if hasattr(config, "eos_token_id") and config.eos_token_id:
            print(f"eos_token_id {config.eos_token_id} from config.json")
            eos_token_id = config.eos_token_id
            eos_token = tokenizer.convert_ids_to_tokens(config.eos_token_id)
        elif hasattr(config, "eos_token") and config.eos_token:
            print(f"eos_token {config.eos_token} from config.json")
            eos_token = config.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)
        else:
            raise ValueError(
                "No available eos_token or eos_token_id, please provide eos_token by params or eos_token_id by config.json"
            )

    if pad_token:
        pad_token = pad_token
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        print(f"pad_token {pad_token} from user input")
    else:
        if hasattr(config, "pad_token_id") and config.pad_token_id:
            print(f"pad_token_id {config.pad_token_id} from config.json")
            pad_token_id = config.pad_token_id
            pad_token = tokenizer.convert_ids_to_tokens(config.pad_token_id)
        elif hasattr(config, "pad_token") and config.pad_token:
            print(f"pad_token {config.pad_token} from config.json")
            pad_token = config.pad_token
            pad_token_id = tokenizer.convert_tokens_to_ids(config.pad_token)
        else:
            print(f"pad_token {eos_token} duplicated from eos_token")
            pad_token = eos_token
            pad_token_id = eos_token_id

    # update tokenizer eos_token and pad_token
    tokenizer.eos_token_id = eos_token_id
    tokenizer.eos_token = eos_token
    tokenizer.pad_token_id = pad_token_id
    tokenizer.pad_token = pad_token

    print(
        f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}"
    )
    print(
        f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}"
    )
    print(tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        load_in_8bit=(quantization == "8bit"),
        load_in_4bit=(quantization == "4bit"),
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if peft_path:
        print("Loading PEFT MODEL...")
        model = PeftModel.from_pretrained(base_model, peft_path)
    else:
        print("Loading Original MODEL...")
        model = base_model

    model.eval()

    print(
        "=======================================MODEL Configs====================================="
    )
    print(model.config)
    print(
        "========================================================================================="
    )
    print(
        "=======================================MODEL Archetecture================================"
    )
    print(model)
    print(
        "========================================================================================="
    )

    return model, tokenizer


def hf_inference(
    model, tokenizer, text_list, args=None, max_new_tokens=512, do_sample=True, **kwargs
):
    """
    transformers models inference by huggingface
    """
    inputs = tokenizer(
        text_list, return_tensors="pt", padding=True, add_special_tokens=False
    ).to("cuda")
    print(
        "================================Prompts and Generations============================="
    )

    outputs = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )

    gen_text = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    for i in range(len(text_list)):
        print("=========" * 10)
        print(f"Prompt:\n{text_list[i]}")
        gen_text[i] = gen_text[i].replace(tokenizer.pad_token, "")
        print(f"Generation:\n{gen_text[i]}")
        sys.stdout.flush()

    return gen_text


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_or_path", type=str, default=None, help="Path to the base model"
    )
    parser.add_argument(
        "--adaptor_path", type=str, default=None, help="Path to the adaptor checkpoint"
    )
    parser.add_argument(
        "--eval_path", type=str, default=None, help="Path to the evaluation file"
    )
    parser.add_argument(
        "--give_expl",
        action="store_true",
        default=None,
        help="Give explanation for the vulnerability as prompt",
    )

    args = parser.parse_args()

    base_model = args.base_model_or_path
    lora_adapter = args.adaptor_path
    eval_path = args.eval_path

    HUMAN_ROLE_START_TAG = "<s>human\n"
    BOT_ROLE_START_TAG = "<s>bot\n"

    localization_instruction = """
    Does the following code have a security vulnerability:

    '''
    public boolean checkUserPassword(String userId, String password) {
    if (StringUtils.isBlank(userId)) {
        MSException.throwException(Translator.get("user_name_is_null"));
    }
    if (StringUtils.isBlank(password)) {
        MSException.throwException(Translator.get("password_is_null"));
    }
    UserExample example = new UserExample();
    example.createCriteria().andIdEqualTo(userId).andPasswordEqualTo(CodingUtil.md5(password));
    return userMapper.countByExample(example) > 0;
    }

    public UserDTO loginLocalMode(String userId, String password) {
    UserDTO user = getLoginUser(userId, Collections.singletonList(UserSource.LOCAL.name()));
    if (user == null) {
        user = getUserDTOByEmail(userId, UserSource.LOCAL.name());
        if (user == null) {
            throw new RuntimeException(Translator.get("password_is_incorrect"));
        }
        userId = user.getId();
    }
    if (!checkUserPassword(userId, password)) {
        throw new RuntimeException(Translator.get("password_is_incorrect"));
    }
    user.setPassword(null);
    return user;
    '''


}
  
    """

    explanation_instruction = """
    What is the impact of the vulnerability in the above code? How would an attacker use this vulnerability?
    """

    if args.give_expl:
        localization_instruction += """
            As context, this code is apart of MeterSphere, an open source continuous testing platform.
            The `checkUserPassword` method is used to check whether the password provided by the user matches the password saved in the database.

            Notice that the code does not perform a check on userId or password length, potentially leading to allocation of too much memory. 
        """
        explanation_instruction = (
            """
            The actual CWE is CWE-770 (allocation of resources without
            limits or throttling) """
            + explanation_instruction
        )

    texts = [localization_instruction, explanation_instruction]
    prompts = [f"{HUMAN_ROLE_START_TAG}{text}{BOT_ROLE_START_TAG}" for text in texts]

    current_directory = os.getcwd()
    model, tokenizer = load_model_tokenizer(
        base_model,
        model_type="",
        peft_path=lora_adapter,
        eos_token="</s>",
        pad_token="<unk>",
    )
    hf_inference(model, tokenizer, prompts, do_sample=True, temperature=0.8)
