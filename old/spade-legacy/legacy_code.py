class Model():
    #         return out
    def _get_rel_loss_no_mask(self, rel, labels):
        # print(rel.shape, labels.shape)
        # ml = labels.shape[-1]
        # rel = rel.view(-1, ml)
        # labels = labels.view(-1, ml)
        labels = labels.type(torch.float)
        losses = self.loss_func(rel, labels)
        return losses

    def _get_rel_loss(self, rel, labels, token_types, n_classes=None):
        # Input: batch * 2 * node * node
        # Input: batch * node * node
        # token_types: batch * node
        fill_value = self.fill_value
        if n_classes is None:
            n_classes = self.n_classes

        # repeat channel on labels
        # labels = labels.unsqueeze(1)
        # labels = torch.cat([labels, 1 - labels], dim=1)
        # print(labels.shape, rel.shape)

        # Mask label part
        mask_top = torch.zeros(
            [1, 1, n_classes, self.config_bert.max_position_embeddings],
            dtype=torch.bool,
            device=rel.device,
        )

        # Mask diagonal
        self_mask = torch.eye(rel.shape[-1], dtype=torch.bool, device=rel.device)
        self_mask = self_mask.unsqueeze(0).unsqueeze(0)
        self_mask = torch.cat([mask_top, self_mask], dim=2)
        # print(self_mask.shape, rel.shape, labels.shape)
        rel = rel.masked_fill(self_mask, fill_value)
        labels = labels.masked_fill(self_mask, fill_value)

        # Mask special tokens
        # Orig size: batch * seq
        special_mask = token_types > 1
        special_mask = special_mask.unsqueeze(1).unsqueeze(-1)
        special_mask = special_mask * special_mask.transpose(-1, -2)
        mask_top = torch.zeros(
            [
                special_mask.shape[0],
                1,
                n_classes,
                self.config_bert.max_position_embeddings,
            ],
            dtype=torch.bool,
            device=rel.device,
        )
        special_mask = torch.cat([mask_top, special_mask], dim=2)
        rel = rel.masked_fill(special_mask, fill_value)
        labels = labels.masked_fill(special_mask, fill_value)

        # labels = torch.broadcast_to(labels, rel.shape)

        # broadcast_to(labels, rel.shape)
        # labels = torch.cat(
        #     [labels, torch.ones_like(labels, device=labels.device)],  #! line break
        #     dim=1,
        # )
        ml = rel.shape[-1]
        return self.loss_func(rel.view(-1, ml), labels.view(-1, ml))

    # def forward(self, batch):
    #     if "text_tokens" in batch:
    #         # Text tokens
    #         batch.pop("text_tokens")
    #     batch = BatchEncoding(batch)
    #     outputs = self.backbone(
    #         input_ids=batch.input_ids,
    #         # bbox=batch.bbox,
    #         attention_mask=batch.attention_mask,
    #     )  # , token_type_ids=token_type_ids)
    #     last_hidden_state = outputs.last_hidden_state
    #     # print(last_hidden_state.shape)
    #     # last_hidden_state = last_hidden_state.transpose(-1, -2).contiguous()
    #     itc_outputs = self.itc_layer(last_hidden_state)  # .transpose(0, 1).contiguous()
    #     itc_outputs = self.act(itc_outputs)
    #     # print(itc_outputs.shape)
    #     last_hidden_state = last_hidden_state.transpose(0, 1).contiguous()
    #     stc_outputs = self.stc_layer(last_hidden_state, last_hidden_state).squeeze(0)
    #     # stc_outputs = self.threshold(stc_outputs)
    #     # itc_outputs = self.threshold(itc_outputs)
    #     # itc_labels = batch.itc_labels
    #     # itc_labels = torch.functional.onehots(itc_labels, self.n_classes)
    #     # batch.itc_labels = self.label_morph(batch.itc_labels)
    #     out = SpadeOutput(
    #         itc_outputs=(itc_outputs),
    #         stc_outputs=(stc_outputs),
    #         attention_mask=batch.attention_mask,
    #         loss=self._get_loss(itc_outputs, stc_outputs, batch),
    #     )
    #     return out

    def _get_loss(self, itc_outputs, stc_outputs, batch):
        itc_loss = self._get_itc_loss(itc_outputs, batch)
        stc_loss = self._get_stc_loss(stc_outputs, batch)
        # print("itc_loss", itc_loss.item(), torch.norm(itc_loss))
        # print("stc_loss", stc_loss.item(), torch.norm(stc_loss))
        # loss = itc_loss + stc_loss
        # loss =

        return itc_loss, stc_loss

    def _get_itc_loss(self, itc_outputs, batch):
        itc_mask = batch.attention_mask
        inv_mask = (1 - itc_mask).bool()
        itc_outputs = itc_outputs.masked_fill(inv_mask.unsqueeze(-1), -1.0)
        itc_outputs = itc_outputs.transpose(-1, -2)
        labels = batch.itc_labels

        return self.itc_loss_func(itc_outputs, labels)

    #     def _get_itc_loss(self, itc_outputs, batch):
    #         itc_mask = batch["are_box_first_tokens"].view(-1).bool()
    #         itc_mask = torch.where(itc_mask)

    #         itc_logits = itc_outputs.view(-1, self.n_classes)
    #         itc_logits = itc_logits[itc_mask]
    #         self.field_map = self.field_map.to(itc_logits.device)
    #         itc_labels = self.field_map[batch["itc_labels"]].view(-1)
    #         itc_labels = itc_labels[itc_mask]

    #         itc_loss = self.loss_func(itc_logits, itc_labels)

    #         return itc_loss

    # def _get_stc_loss(self, stc_outputs, batch):
    #     labels = batch.stc_labels
    #     outputs = stc_outputs.transpose(0, 1)
    #     # print(outputs.shape)
    #     # mask_x, mask_y = torch.where(batch.attention_mask.bool())
    #     nrel = labels.shape[1]
    #     # losses = [
    #     return self.loss_func(outputs, labels)

    # ]
    # return sum(losses) / nrel

    #         inv_atm = (1 - batch.attention_mask)[:, None, :]

    #         labels = labels.transpose(0, 1).masked_fill(inv_atm, -10000.0)
    #         outputs = labels.masked_fill(inv_atm, -10000.0)

    #         outputs = outputs.transpose(0, 1)
    #         labels = labels.transpose(0, 1)

    # nrel = outputs.shape[1]
    # return loss

    # bsize = outputs.shape[0]
    # loss = [
    #     self.loss_func(outputs[:, i, :, :], labels[:, i, :, :])
    #     for i in range(nrel)]
    # loss = 0
    # for b in range(bsize):
    #     for i in range(nrel):
    #         loss += self.loss_func(outputs[b, i, :, :], labels[b, i, :, :])
    # return sum(loss)
    # return sum([self.loss_func(stc_outputs[:, i, :, :], batch.stc_labels[:, i, :, :])
    #             for i in range(2)])

    def _get_stc_loss(self, stc_outputs, batch):
        invalid_token_mask = 1 - batch["attention_mask"]

        bsz, max_seq_length = invalid_token_mask.shape
        device = invalid_token_mask.device

        # invalid_token_mask = torch.cat(
        #     [inv_attention_mask, torch.zeros([bsz, 1]).to(device)], axis=1
        # ).bool()
        stc_outputs.masked_fill_(invalid_token_mask[:, None, :].bool(), -1.0)

        self_token_mask = torch.eye(max_seq_length, max_seq_length).to(device).bool()
        stc_outputs.masked_fill_(self_token_mask[None, :, :].bool(), -1.0)

        stc_mask = batch["attention_mask"].view(-1).bool()
        stc_mask = torch.where(stc_mask)

        stc_logits = stc_outputs.view(-1, max_seq_length)
        stc_logits = stc_logits[stc_mask]

        stc_labels = batch["stc_labels"]
        stc_labels = stc_labels.flatten()
        stc_labels = stc_labels[stc_mask]
        # print("labels", stc_labels.shape)
        # print("logits", stc_logits.shape)
        stc_labels = torch.broadcast_to(stc_labels.unsqueeze(1), stc_logits.shape)

        # print("labels", stc_labels.shape)
        # print("logits", stc_logits.shape)
        stc_loss = self.loss_func(stc_logits, stc_labels)

        return stc_loss

    def true_adj_single(itc_out, stc_out, attention_mask):
        idx = attention_mask.diff(dim=-1).argmin(dim=-1) + 1
        true_itc_out = itc_out[:idx]
        true_stc_out = stc_out[:idx, :idx]
        return torch.cat([true_itc_out, true_stc_out], dim=-1).transpose(0, 1)


