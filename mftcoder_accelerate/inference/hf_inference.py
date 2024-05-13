# -*- coding: utf-8 -*-
# @author Chaoyu Chen
# @date 2024/1/4
# @module hf_inference.py

import os
import sys
import torch
import textwrap
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel
import argparse
import json
import jsonlines



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
    parser.add_argument("--base_model_or_path", type=str, default=None)
    parser.add_argument("--adaptor_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)

    args = parser.parse_args()

    base_model = args.base_model_or_path
    lora_adapter = args.adaptor_path
    eval_path = args.eval_path

    HUMAN_ROLE_START_TAG = "<s>human\n"
    BOT_ROLE_START_TAG = "<s>bot\n"

    localization_instruction = """
    Find which lines in following code has a security vulnerability:
    /* -*- mode: c; c-basic-offset: 4; indent-tabs-mode: nil -*- */\n#include \"k5-int.h\"\n#include <kadm5/admin.h>\n#include <syslog.h>\n#include <adm_proto.h>  /* krb5_klog_syslog */\n#include <stdio.h>\n#include <errno.h>\n\n#include \"kadm5/server_internal.h\" /* XXX for kadm5_server_handle_t */\n\n#include \"misc.h\"\n\n#ifndef GETSOCKNAME_ARG3_TYPE\n#define GETSOCKNAME_ARG3_TYPE int\n#endif\n\n#define RFC3244_VERSION 0xff80\n\nstatic krb5_error_code\nprocess_chpw_request(krb5_context context, void *server_handle, char *realm,\n                     krb5_keytab keytab, const krb5_fulladdr *local_faddr,\n                     const krb5_fulladdr *remote_faddr, krb5_data *req,\n                     krb5_data *rep)\n{\n    krb5_error_code ret;\n    char *ptr;\n    unsigned int plen, vno;\n    krb5_data ap_req, ap_rep = empty_data();\n    krb5_data cipher = empty_data(), clear = empty_data();\n    krb5_auth_context auth_context = NULL;\n    krb5_principal changepw = NULL;\n    krb5_principal client, target = NULL;\n    krb5_ticket *ticket = NULL;\n    krb5_replay_data replay;\n    krb5_error krberror;\n    int numresult;\n    char strresult[1024];\n    char *clientstr = NULL, *targetstr = NULL;\n    const char *errmsg = NULL;\n    size_t clen;\n    char *cdots;\n    struct sockaddr_storage ss;\n    socklen_t salen;\n    char addrbuf[100];\n    krb5_address *addr = remote_faddr->address;\n\n    *rep = empty_data();\n\n    if (req->length < 4) {\n        /* either this, or the server is printing bad messages,\n           or the caller passed in garbage */\n        ret = KRB5KRB_AP_ERR_MODIFIED;\n        numresult = KRB5_KPASSWD_MALFORMED;\n        strlcpy(strresult, \"Request was truncated\", sizeof(strresult));\n        goto chpwfail;\n    }\n\n    ptr = req->data;\n\n    /* verify length */\n\n    plen = (*ptr++ & 0xff);\n    plen = (plen<<8) | (*ptr++ & 0xff);\n\n    if (plen != req->length) {\n        ret = KRB5KRB_AP_ERR_MODIFIED;\n        numresult = KRB5_KPASSWD_MALFORMED;\n        strlcpy(strresult, \"Request length was inconsistent\",\n                sizeof(strresult));\n        goto chpwfail;\n    }\n\n    /* verify version number */\n\n    vno = (*ptr++ & 0xff) ;\n    vno = (vno<<8) | (*ptr++ & 0xff);\n\n    if (vno != 1 && vno != RFC3244_VERSION) {\n        ret = KRB5KDC_ERR_BAD_PVNO;\n        numresult = KRB5_KPASSWD_BAD_VERSION;\n        snprintf(strresult, sizeof(strresult),\n                 \"Request contained unknown protocol version number %d\", vno);\n        goto chpwfail;\n    }\n\n    /* read, check ap-req length */\n\n    ap_req.length = (*ptr++ & 0xff);\n    ap_req.length = (ap_req.length<<8) | (*ptr++ & 0xff);\n\n    if (ptr + ap_req.length >= req->data + req->length) {\n        ret = KRB5KRB_AP_ERR_MODIFIED;\n        numresult = KRB5_KPASSWD_MALFORMED;\n        strlcpy(strresult, \"Request was truncated in AP-REQ\",\n                sizeof(strresult));\n        goto chpwfail;\n    }\n\n    /* verify ap_req */\n\n    ap_req.data = ptr;\n    ptr += ap_req.length;\n\n    ret = krb5_auth_con_init(context, &auth_context);\n    if (ret) {\n        numresult = KRB5_KPASSWD_HARDERROR;\n        strlcpy(strresult, \"Failed initializing auth context\",\n                sizeof(strresult));\n        goto chpwfail;\n    }\n\n    ret = krb5_auth_con_setflags(context, auth_context,\n                                 KRB5_AUTH_CONTEXT_DO_SEQUENCE);\n    if (ret) {\n        numresult = KRB5_KPASSWD_HARDERROR;\n        strlcpy(strresult, \"Failed initializing auth context\",\n                sizeof(strresult));\n        goto chpwfail;\n    }\n\n    ret = krb5_build_principal(context, &changepw, strlen(realm), realm,\n                               \"kadmin\", \"changepw\", NULL);\n    if (ret) {\n        numresult = KRB5_KPASSWD_HARDERROR;\n        strlcpy(strresult, \"Failed building kadmin/changepw principal\",\n                sizeof(strresult));\n        goto chpwfail;\n    }\n\n    ret = krb5_rd_req(context, &auth_context, &ap_req, changepw, keytab,\n                      NULL, &ticket);\n\n    if (ret) {\n        numresult = KRB5_KPASSWD_AUTHERROR;\n        strlcpy(strresult, \"Failed reading application request\",\n                sizeof(strresult));\n        goto chpwfail;\n    }\n\n    /* construct the ap-rep */\n\n    ret = krb5_mk_rep(context, auth_context, &ap_rep);\n    if (ret) {\n        numresult = KRB5_KPASSWD_AUTHERROR;\n        strlcpy(strresult, \"Failed replying to application request\",\n                sizeof(strresult));\n        goto chpwfail;\n    }\n\n    /* decrypt the ChangePasswdData */\n\n    cipher.length = (req->data + req->length) - ptr;\n    cipher.data = ptr;\n\n    /*\n     * Don't set a remote address in auth_context before calling krb5_rd_priv,\n     * so that we can work against clients behind a NAT.  Reflection attacks\n     * aren't a concern since we use sequence numbers and since our requests\n     * don't look anything like our responses.  Also don't set a local address,\n     * since we don't know what interface the request was received on.\n     */\n\n    ret = krb5_rd_priv(context, auth_context, &cipher, &clear, &replay);\n    if (ret) {\n        numresult = KRB5_KPASSWD_HARDERROR;\n        strlcpy(strresult, \"Failed decrypting request\", sizeof(strresult));\n        goto chpwfail;\n    }\n\n
    \n
    """

    type_instruction = """
    Which type of CWE vulnerability is present in the above code?\n
    """

    explanation_instruction = """
    What is the impact of the vulnerability in the above code?\n
    """

    texts = [localization_instruction, type_instruction, explanation_instruction]

    prompts = [f"{HUMAN_ROLE_START_TAG}{text}{BOT_ROLE_START_TAG}" for text in texts]

    # read json from file
    with jsonlines.open(eval_path, 'r') as jsonl_f:
        data = [obj for obj in jsonl_f]

    for d in data:
        chat_rounds = d["chat_rounds"]
        question = chat_rounds[1]["content"]
        answer = chat_rounds[2]["content"]
        print(question)
        
        print(answer)
        break
    
    current_directory = os.getcwd()
    # model, tokenizer = load_model_tokenizer(
    #     base_model,
    #     model_type="",
    #     peft_path=lora_adapter,
    #     eos_token="</s>",
    #     pad_token="<unk>",
    # )
    # hf_inference(model, tokenizer, prompts, do_sample=True, temperature=0.8)
